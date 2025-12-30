
import cv2
import numpy as np
import time

def open_camera(preferred_index=0):
    """Try opening the camera with multiple backends (Windows-friendly)."""
    cap = cv2.VideoCapture(preferred_index)
    if cap is not None and cap.isOpened():
        return cap, preferred_index

    cap = cv2.VideoCapture(preferred_index, cv2.CAP_DSHOW)
    if cap is not None and cap.isOpened():
        return cap, preferred_index

    cap = cv2.VideoCapture(preferred_index, cv2.CAP_MSMF)
    if cap is not None and cap.isOpened():
        return cap, preferred_index

    for idx in [1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap, idx

    return None, None

def clamp_roi(x, y, w, h, img_shape):
    """Ensure ROI stays within the image bounds and has at least 1x1 size."""
    H, W = img_shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def draw_axis_aligned_box(img, x1, y1, x2, y2, margin=3, color=(255, 0, 0), thickness=2):
    """Axis-aligned rectangle around a line segment from (x1,y1) to (x2,y2)."""
    xmin = min(x1, x2) - margin
    ymin = min(y1, y2) - margin
    xmax = max(x1, x2) + margin
    ymax = max(y1, y2) + margin
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

# ----- Mouse interaction helpers -----
HANDLE_SIZE = 8
HANDLE_HIT_RADIUS = 12

def point_in_rect(px, py, rx, ry, rw, rh):
    return rx <= px <= rx + rw and ry <= py <= ry + rh

def near_point(px, py, qx, qy, tol=HANDLE_HIT_RADIUS):
    return abs(px - qx) <= tol and abs(py - qy) <= tol

class ROIController:
    """
    Manages mouse interactions for dragging and resizing the ROI.
    - Left-click inside ROI: drag move.
    - Left-click near a corner handle: resize.
    """

    def __init__(self, x, y, w, h, img_shape):
        self.x, self.y, self.w, self.h = clamp_roi(x, y, w, h, img_shape)
        self.dragging = False
        self.resizing = False
        self.resize_anchor = None
        self.drag_offset = (0, 0)
        self.img_shape = img_shape

    def corners(self):
        tl = (self.x, self.y)
        tr = (self.x + self.w, self.y)
        bl = (self.x, self.y + self.h)
        br = (self.x + self.w, self.y + self.h)
        return {"tl": tl, "tr": tr, "bl": bl, "br": br}

    def handle_hit(self, mx, my):
        cs = self.corners()
        for name, (cx, cy) in cs.items():
            if near_point(mx, my, cx, cy, HANDLE_HIT_RADIUS):
                return name
        return None

    def on_mouse(self, event, mx, my, flags, param):
        """Mouse callback to update ROI based on interactions."""
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self.handle_hit(mx, my)
            if hit is not None:
                self.resizing = True
                self.resize_anchor = hit
                self.dragging = False
                return

            if point_in_rect(mx, my, self.x, self.y, self.w, self.h):
                self.dragging = True
                self.resizing = False
                self.resize_anchor = None
                self.drag_offset = (mx - self.x, my - self.y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                nx = mx - self.drag_offset[0]
                ny = my - self.drag_offset[1]
                self.x, self.y, self.w, self.h = clamp_roi(nx, ny, self.w, self.h, self.img_shape)
            elif self.resizing and self.resize_anchor:
                if self.resize_anchor == 'tl':
                    nx = min(mx, self.x + self.w - 1)
                    ny = min(my, self.y + self.h - 1)
                    new_w = (self.x + self.w) - nx
                    new_h = (self.y + self.h) - ny
                    self.x, self.y, self.w, self.h = clamp_roi(nx, ny, new_w, new_h, self.img_shape)
                elif self.resize_anchor == 'tr':
                    nx = self.x
                    ny = min(my, self.y + self.h - 1)
                    new_w = max(1, mx - self.x)
                    new_h = (self.y + self.h) - ny
                    self.x, self.y, self.w, self.h = clamp_roi(nx, ny, new_w, new_h, self.img_shape)
                elif self.resize_anchor == 'bl':
                    nx = min(mx, self.x + self.w - 1)
                    ny = self.y
                    new_w = (self.x + self.w) - nx
                    new_h = max(1, my - self.y)
                    self.x, self.y, self.w, self.h = clamp_roi(nx, ny, new_w, new_h, self.img_shape)
                elif self.resize_anchor == 'br':
                    nx = self.x
                    ny = self.y
                    new_w = max(1, mx - self.x)
                    new_h = max(1, my - self.y)
                    self.x, self.y, self.w, self.h = clamp_roi(nx, ny, new_w, new_h, self.img_shape)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
            self.resize_anchor = None

    def draw(self, img):
        """Draw ROI box and corner handles."""
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 0, 255), 2)
        for (cx, cy) in self.corners().values():
            cv2.rectangle(img,
                          (cx - HANDLE_SIZE // 2, cy - HANDLE_SIZE // 2),
                          (cx + HANDLE_SIZE // 2, cy + HANDLE_SIZE // 2),
                          (255, 0, 255), -1)

# ---- Vertical filtering helpers ----
def line_angle_deg(x1, y1, x2, y2):
    """
    Returns the line angle in degrees mapped to [0, 180).
    0° is horizontal (left->right), 90° is vertical.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    angle = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
    return angle

def is_near_vertical(x1, y1, x2, y2, tol_deg=10):
    """
    True if the line segment angle is within tol_deg of 90° (vertical).
    """
    angle = line_angle_deg(x1, y1, x2, y2)
    if angle is None:
        return False
    return abs(angle - 90.0) <= tol_deg

def main():
    cap, cam_idx = open_camera(preferred_index=0)
    if cap is None:
        print("[error] Could not open any camera. Check privacy settings or close apps using the camera.")
        print("[hint] Windows: Settings → Privacy & security → Camera → Allow apps to access your camera.")
        return
    print(f"[info] Using camera index: {cam_idx}")

    # Set capture resolution & FPS (lower if preview lags)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Detection params (box all lines regardless of size)
    CANNY_LOW = 30
    CANNY_HIGH = 100
    HOUGH_THRESHOLD = 70  # will be adjusted by mouse wheel
    MIN_LINE_LENGTH = 3
    MAX_LINE_GAP = 12

    HOUGH_MIN = 10   # clamp low end to avoid too much noise
    HOUGH_MAX = 200  # clamp high end to avoid missing everything
    HOUGH_STEP = 5   # per mouse wheel notch

    # Vertical filter settings
    IGNORE_VERTICAL = True
    TOL_VERTICAL_DEG = 10         # initial tolerance around 90°
    TOL_VERTICAL_DEG_MIN = 0
    TOL_VERTICAL_DEG_MAX = 45
    TOL_VERTICAL_STEP = 1

    window_name = "Drag ROI + Mouse Wheel (Hough threshold) | Lines view"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Read an initial frame
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[error] Could not read initial frame.")
        cap.release()
        return
    H, W = frame.shape[:2]

    # ROI starts centered covering ~50% of the frame
    roi = ROIController(x=W//4, y=H//4, w=W//2, h=H//2, img_shape=frame.shape)

    # Shared state to let mouse callback adjust HOUGH_THRESHOLD
    state = {"HOUGH_THRESHOLD": HOUGH_THRESHOLD}

    # Mouse callback combining ROI interactions + mouse wheel adjustment
    def mouse_cb(event, mx, my, flags, param):
        # First pass ROI interactions to controller
        roi.on_mouse(event, mx, my, flags, param)

        # Handle mouse wheel for HOUGH_THRESHOLD
        if event == cv2.EVENT_MOUSEWHEEL:
            # On many platforms: flags > 0 means wheel up, flags < 0 means wheel down
            if flags > 0:
                state["HOUGH_THRESHOLD"] = min(HOUGH_MAX, state["HOUGH_THRESHOLD"] + HOUGH_STEP)
            else:
                state["HOUGH_THRESHOLD"] = max(HOUGH_MIN, state["HOUGH_THRESHOLD"] - HOUGH_STEP)

    cv2.setMouseCallback(window_name, mouse_cb)

    # Arrow key codes (Windows via waitKeyEx)
    VK_LEFT  = 2424832
    VK_RIGHT = 2555904
    VK_UP    = 2490368
    VK_DOWN  = 2621440

    move_step = 10
    size_w_step = 10
    size_h_step = 6

    print("[info] Controls:")
    print("   • Mouse: click-drag inside ROI to move; drag corner squares to resize; scroll wheel adjusts Hough threshold")
    print("   • Keys: +/- resize ROI, arrows/WASD move ROI, s snapshot, q quit")
    print("   • Vertical filter: [ decreases tol, ] increases tol; v toggles ignore-vertical on/off")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[warn] Failed to read frame. Retrying...")
            time.sleep(0.05)
            continue

        # Ensure ROI stays valid if frame size changes
        roi.x, roi.y, roi.w, roi.h = clamp_roi(roi.x, roi.y, roi.w, roi.h, frame.shape)

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Crop to ROI
        roi_gray = gray[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]

        # Edges in ROI
        edges_roi = cv2.Canny(roi_gray, CANNY_LOW, CANNY_HIGH, apertureSize=3)

        # HoughLinesP on ROI using current threshold from mouse wheel state
        HOUGH_THRESHOLD_CURR = state["HOUGH_THRESHOLD"]
        lines_p = cv2.HoughLinesP(
            edges_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD_CURR,
            minLineLength=MIN_LINE_LENGTH,
            maxLineGap=MAX_LINE_GAP
        )

        output = frame.copy()

        # Draw ROI and handles
        roi.draw(output)

        # Draw every detected line + box; count them (after vertical filtering)
        box_count = 0
        filtered_count = 0
        if lines_p is not None and len(lines_p) > 0:
            for x1, y1, x2, y2 in lines_p[:, 0]:
                # Skip near-vertical lines if enabled
                if IGNORE_VERTICAL and is_near_vertical(x1, y1, x2, y2, tol_deg=TOL_VERTICAL_DEG):
                    filtered_count += 1
                    continue

                gx1, gy1 = x1 + roi.x, y1 + roi.y
                gx2, gy2 = x2 + roi.x, y2 + roi.y
                cv2.line(output, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                draw_axis_aligned_box(output, gx1, gy1, gx2, gy2, margin=3, color=(255, 0, 0), thickness=2)
                box_count += 1

        # Overlay counts and current HOUGH_THRESHOLD
        cv2.putText(output, f"Boxes in ROI: {box_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"Hough threshold: {HOUGH_THRESHOLD_CURR}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)

        # Vertical filter overlay
        cv2.putText(
            output,
            f"Ignore vertical: {'ON' if IGNORE_VERTICAL else 'OFF'} | tol ±{TOL_VERTICAL_DEG}° | filtered: {filtered_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2, cv2.LINE_AA
        )
        cv2.putText(output, "Wheel: up stricter / down more sensitive", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(output, "[ ] tol, v toggle vertical filter", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)

        # Show single view (left: annotated frame)
        display_frame = cv2.resize(output, (800, 600))
        cv2.imshow(window_name, display_frame)

        # Use waitKeyEx for extended keys (arrows)
        key = cv2.waitKeyEx(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # NOTE: 's' also appears in WASD; snapshot takes precedence.
            ts = int(time.time())
            out_name = f"snapshot_{ts}.jpg"
            ok = cv2.imwrite(out_name, output)
            print(f"[info] Snapshot saved: {out_name}" if ok else f"[error] Failed to save snapshot: {out_name}")

        # Arrow keys
        elif key == VK_LEFT:
            roi.x -= move_step
        elif key == VK_RIGHT:
            roi.x += move_step
        elif key == VK_UP:
            roi.y -= move_step
        elif key == VK_DOWN:
            roi.y += move_step

        # +/- resize (handle shifted/unshifted)
        elif key in (ord('+'), ord('=')):
            roi.w += size_w_step
            roi.h += size_h_step
        elif key in (ord('-'), ord('_')):
            roi.w -= size_w_step
            roi.h -= size_h_step

        # WASD fallback (note: 's' is snapshot above)
        elif key == ord('a'):
            roi.x -= move_step
        elif key == ord('d'):
            roi.x += move_step
        elif key == ord('w'):
            roi.y -= move_step
        elif key == ord('s'):
            roi.y += move_step

        # Vertical filter tuning
        elif key == ord('['):
            TOL_VERTICAL_DEG = max(TOL_VERTICAL_DEG_MIN, TOL_VERTICAL_DEG - TOL_VERTICAL_STEP)
        elif key == ord(']'):
            TOL_VERTICAL_DEG = min(TOL_VERTICAL_DEG_MAX, TOL_VERTICAL_DEG + TOL_VERTICAL_STEP)
        elif key == ord('v'):
            IGNORE_VERTICAL = not IGNORE_VERTICAL

        # Re-clamp after any change
        roi.x, roi.y, roi.w, roi.h = clamp_roi(roi.x, roi.y, roi.w, roi.h, frame.shape)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
