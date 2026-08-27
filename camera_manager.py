#!/usr/bin/env python3
"""
EVA ROBOT - CAMERA MANAGER (USB via OpenCV, PiCam via Picamera2 se disponível)
- Auto-detect de /dev/video*
- Switch robusto com fallback
- Rotação aplicada SOMENTE na PiCam
"""

import time
import threading
from enum import Enum
from typing import Optional, List, Tuple

import cv2
import numpy as np


class CameraType(Enum):
    USB = "usb"
    PICAM = "picam"


class CameraManager:
    def __init__(
        self,
        picam_id: int = 0,
        usb_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        rotate_picam: bool = True,
        picam_rotation=cv2.ROTATE_90_CLOCKWISE,
        flip_usb: bool = False,
        usb_flip_code: int = 1,  # 1=hflip,0=vflip,-1=both
    ):
        self.picam_id = int(picam_id)
        self.usb_id = int(usb_id)

        self.width = int(width)
        self.height = int(height)
        self.fps = max(5, int(fps))

        self.rotate_picam = bool(rotate_picam)
        self.picam_rotation = picam_rotation

        self.flip_usb = bool(flip_usb)
        self.usb_flip_code = int(usb_flip_code)

        self.active_camera_type = CameraType.USB

        # OpenCV capture (USB ou fallback)
        self.cap: Optional[cv2.VideoCapture] = None

        # Picamera2 (se existir)
        self.picam2 = None
        self.picam2_started = False

        self.frame: Optional[np.ndarray] = None
        self.last_good_frame: Optional[np.ndarray] = None

        self.frame_lock = threading.Lock()
        self.cap_lock = threading.Lock()

        self.running = False
        self.switching = False
        self.thread: Optional[threading.Thread] = None

        print("📷 CameraManager inicializado")

    # -------------------------
    # utils detect
    # -------------------------
    def _detect_opencv_devices(self, max_index: int = 6) -> List[int]:
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            ok = cap.isOpened()
            cap.release()
            if ok:
                found.append(i)
        return found

    def _open_opencv(self, device_id: int) -> bool:
        cap = cv2.VideoCapture(int(device_id), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        # MJPG explícito -- ACHADO EM USO REAL: sem isso, alguns
        # webcams USB (comum em V4L2) aceitam abrir (isOpened()=True) e
        # aceitam os cap.set() de resolução/fps sem erro, mas o
        # cap.read() nunca traz frame de verdade -- o driver não
        # consegue negociar o formato bruto (YUYV) na resolução pedida,
        # mas MJPG geralmente funciona na mesma resolução porque é
        # comprimido pela própria câmera antes de sair pela USB. Setar
        # isto ANTES de largura/altura/fps porque em alguns drivers a
        # ordem importa (fourcc primeiro, depois negociar resolução
        # dentro daquele formato).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Aquecimento que CHECA de verdade se leu frame -- ANTES, os 3
        # cap.read() de aquecimento não checavam o retorno (ok, frame),
        # então um device que abre mas nunca entrega frame passava pelo
        # aquecimento "com sucesso" e só ia falhar de verdade minutos
        # depois, silenciosamente, dentro de _capture_loop -- sem log
        # nenhum explicando por quê (ver eva_command_server._video_loop,
        # que precisou ganhar diagnóstico à parte só pra expor esse
        # sintoma do lado de fora). Agora, se o aquecimento falhar com
        # MJPG, tenta de novo SEM fourcc forçado (alguns drivers só
        # aceitam o formato bruto padrão) antes de desistir.
        aqueceu = False
        for tentativa in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                aqueceu = True
                break
            time.sleep(0.05)

        if not aqueceu:
            print(f"⚠️  Câmera USB (device {device_id}) abriu mas não entregou frame "
                  f"com MJPG -- tentando sem fourcc forçado...")
            cap.set(cv2.CAP_PROP_FOURCC, 0)  # remove força de fourcc, deixa driver escolher
            for tentativa in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    aqueceu = True
                    break
                time.sleep(0.05)

        if not aqueceu:
            print(f"❌ Câmera USB (device {device_id}) abriu mas cap.read() nunca "
                  f"trouxe frame em nenhum dos dois formatos -- provavelmente resolução "
                  f"{self.width}x{self.height}@{self.fps}fps não suportada por este "
                  f"dispositivo, ou outro processo já está usando a câmera.")
            cap.release()
            return False

        self.cap = cap
        return True

    def _close_opencv(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _open_picam2(self) -> bool:
        try:
            from picamera2 import Picamera2
        except Exception:
            return False

        try:
            if self.picam2 is None:
                self.picam2 = Picamera2()

            # configuração simples
            cfg = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            self.picam2_started = True

            # warmup
            for _ in range(3):
                _ = self.picam2.capture_array()
                time.sleep(0.02)

            return True
        except Exception as e:
            print(f"❌ Falha Picamera2: {e}")
            self._close_picam2()
            return False

    def _close_picam2(self):
        if self.picam2 is not None:
            try:
                if self.picam2_started:
                    self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
        self.picam2 = None
        self.picam2_started = False

    # -------------------------
    # start/stop
    # -------------------------
    def start(self) -> bool:
        self.running = True

        # auto-detect para evitar “USB=1” quando só existe /dev/video0
        devs = self._detect_opencv_devices()
        if devs:
            # assume primeiro como USB (é o mais comum)
            self.usb_id = devs[0]

        # tenta USB (opencv)
        if self._open_opencv(self.usb_id):
            self.active_camera_type = CameraType.USB
        else:
            # tenta PiCam via Picamera2
            if self._open_picam2():
                self.active_camera_type = CameraType.PICAM
            else:
                # tenta opencv em outros índices (fallback)
                for d in devs[1:]:
                    if self._open_opencv(d):
                        self.usb_id = d
                        self.active_camera_type = CameraType.USB
                        break
                else:
                    print("❌ Nenhuma câmera disponível")
                    self.running = False
                    return False

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"✅ CameraManager start ({self.active_camera_type.value})")
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        with self.cap_lock:
            self._close_opencv()
            self._close_picam2()

        print("✅ CameraManager stop")

    # -------------------------
    # switching
    # -------------------------
    def switch_camera(self, camera_type: Optional[CameraType] = None):
        if camera_type is None:
            camera_type = CameraType.PICAM if self.active_camera_type == CameraType.USB else CameraType.USB

        if camera_type == self.active_camera_type:
            return

        print(f"🔁 Alternando câmera para {camera_type.value.upper()}")
        self.switching = True
        try:
            with self.cap_lock:
                # fecha tudo antes de abrir
                self._close_opencv()
                self._close_picam2()

                time.sleep(0.2)

                if camera_type == CameraType.PICAM:
                    # preferir Picamera2
                    if not self._open_picam2():
                        # fallback: tenta opencv em índices disponíveis
                        devs = self._detect_opencv_devices()
                        ok = False
                        for d in devs:
                            if self._open_opencv(d):
                                ok = True
                                break
                        if not ok:
                            print("❌ Falha ao abrir PICAM (e sem fallback)")
                            return
                else:
                    # USB via OpenCV: garante id válido
                    devs = self._detect_opencv_devices()
                    if self.usb_id not in devs and devs:
                        self.usb_id = devs[0]
                    if not self._open_opencv(self.usb_id):
                        print(f"❌ Falha ao abrir USB (device {self.usb_id})")
                        return

                self.active_camera_type = camera_type

        finally:
            self.switching = False

    # -------------------------
    # capture loop
    # -------------------------
    def _capture_loop(self):
        interval = 1.0 / float(self.fps)
        falhas_consecutivas = 0
        ultimo_log_falha = 0.0

        while self.running:
            if self.switching:
                time.sleep(0.01)
                continue

            frame = None
            with self.cap_lock:
                if self.active_camera_type == CameraType.PICAM and self.picam2_started and self.picam2 is not None:
                    try:
                        frame = self.picam2.capture_array()  # RGB
                        # Picamera2 -> vem RGB; OpenCV espera BGR para putText/encode:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception:
                        frame = None
                else:
                    cap = self.cap
                    if cap is not None:
                        ok, f = cap.read()
                        if ok and f is not None:
                            frame = f

            if frame is not None:
                falhas_consecutivas = 0
                if self.active_camera_type == CameraType.PICAM and self.rotate_picam:
                    frame = cv2.rotate(frame, self.picam_rotation)

                if self.active_camera_type == CameraType.USB and self.flip_usb:
                    frame = cv2.flip(frame, self.usb_flip_code)

                with self.frame_lock:
                    self.frame = frame
                    self.last_good_frame = frame
            else:
                # Log com throttle -- diagnóstico mais próximo da causa
                # que o de eva_command_server._video_loop (aquele só vê
                # "sem frame pra codificar", este mostra se é o
                # cap.read() em si que está falhando -- útil se o
                # problema for intermitente, câmera funciona um tempo e
                # para no meio, em vez de nunca funcionar desde o início
                # (esse segundo caso já é pego no aquecimento de
                # _open_opencv, que agora checa de verdade).
                falhas_consecutivas += 1
                agora = time.time()
                if agora - ultimo_log_falha > 5.0:
                    print(f"⚠️  _capture_loop sem frame há {falhas_consecutivas} ciclos "
                          f"(câmera ativa: {self.active_camera_type.value})")
                    ultimo_log_falha = agora

            time.sleep(interval)

    # -------------------------
    # frame API
    # -------------------------
    def get_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            if self.last_good_frame is not None:
                return self.last_good_frame.copy()
            return None

    def get_frame_encoded(self, quality: int = 70) -> Optional[bytes]:
        frame = self.get_frame()
        if frame is None:
            return None

        label = self.active_camera_type.value.upper()
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return None
        return buf.tobytes()

    def get_active_camera_type(self) -> CameraType:
        return self.active_camera_type

    def get_status(self) -> dict:
        return {
            "active_camera": self.active_camera_type.value,
            "running": self.running,
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "switching": self.switching,
            "usb_id": self.usb_id,
            "picam_id": self.picam_id,
            "picam2": bool(self.picam2_started),
        }
