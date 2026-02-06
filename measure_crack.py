import cv2
import numpy as np
import os
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
IMAGE_PATH = "photo.png"   # <-- Put your image filename here
POINT_RADIUS = 5
POINT_COLOR = (0, 200, 255)  # Orange-ish
LINE_COLOR = (0, 255, 0)     # Green
TEXT_COLOR = (50, 50, 255)   # Red-ish
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Zoom/Pan config
ZOOM_STEP = 1.25
MIN_SCALE = 0.1
MAX_SCALE = 20.0
BG_COLOR = (30, 30, 30)  # background outside image

# Auto-calibrate from the first drawn segment
AUTO_CALIBRATE_FROM_FIRST = True
KNOWN_FIRST_LENGTH = 33.1875  # inches
KNOWN_UNIT = "in"

# ---------------------------
# State
# ---------------------------
segments = []            # list of ((x1,y1), (x2,y2))
first_point = None       # current "start" point of a segment (world coords)
pixel_per_unit = None    # pixels per unit (e.g., px/in)
unit_label = None        # e.g., "in"
calibrating = False
calib_points_screen = [] # for optional manual calibration clicks in screen coords

# View transform (screen <-> image)
scale = 1.0
offset = np.array([0.0, 0.0], dtype=float)  # translation in screen pixels
panning = False
pan_start = None

# For hover preview
last_mouse_xy = (0, 0)

# Direction of the first segment (unit vector)
first_seg_unit = None         # np.array([ux, uy]) or None

# ---------------------------
# Helpers
# ---------------------------
def distance(p1, p2):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return float(np.linalg.norm(p2 - p1))

def format_length_only_units(px_len):
    """Return only the real-world length string (no pixels). Falls back to px if not calibrated."""
    if pixel_per_unit and unit_label:
        real = px_len / pixel_per_unit
        return f"{real:.3f} {unit_label}"
    else:
        return f"{px_len:.2f} px"

def world_to_screen(pt):
    """Image coords -> screen coords"""
    return (int(pt[0] * scale + offset[0]), int(pt[1] * scale + offset[1]))

def screen_to_world(x, y):
    """Screen coords -> image coords (float)"""
    return ((x - offset[0]) / scale, (y - offset[1]) / scale)

def clamp_scale(s):
    return max(MIN_SCALE, min(MAX_SCALE, s))

def fit_to_window(img_shape, win_name):
    """Set scale+offset to fit the image into the current window."""
    global scale, offset
    h, w = img_shape[:2]
    try:
        # OpenCV 4.5+: returns (x,y,w,h)
        _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
    except Exception:
        win_w, win_h = w, h
    if win_w <= 0 or win_h <= 0:
        win_w, win_h = w, h
    # Leave small margins
    margin = 40
    sx = (win_w - margin) / w
    sy = (win_h - margin) / h
    scale = clamp_scale(min(sx, sy))
    # Center the image
    disp_w = w * scale
    disp_h = h * scale
    offset[:] = [(win_w - disp_w) / 2.0, (win_h - disp_h) / 2.0]

def cumulative_length_px():
    return sum(distance(p1, p2) for (p1, p2) in segments)

def set_first_direction(seg):
    """Compute and store the unit direction for the first segment."""
    global first_seg_unit
    (a, b) = seg
    v = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    n = np.linalg.norm(v)
    if n > 1e-9:
        first_seg_unit = v / n
    else:
        first_seg_unit = None

def snap_to_parallel(base_point_world, target_world):
    """
    Given a base point (first click) and current target (mouse or click) in world coords,
    return the target snapped to the line through base point with direction first_seg_unit.
    """
    if first_seg_unit is None:
        return target_world
    base = np.array(base_point_world, dtype=float)
    t = np.array(target_world, dtype=float)
    delta = t - base
    proj_len = float(delta[0] * first_seg_unit[0] + delta[1] * first_seg_unit[1])  # dot
    snapped = base + proj_len * first_seg_unit
    return (snapped[0], snapped[1])

def draw_segments_on_canvas(canvas, draw_hover_preview=None):
    """Draw all segments and optional hover preview (in screen coords)."""
    # Draw completed segments
    for seg_idx, (p1, p2) in enumerate(segments, start=1):
        s1 = world_to_screen(p1)
        s2 = world_to_screen(p2)
        cv2.line(canvas, s1, s2, LINE_COLOR, THICKNESS, cv2.LINE_AA)
        # Endpoints
        for spt in (s1, s2):
            cv2.circle(canvas, spt, max(2, int(POINT_RADIUS)), POINT_COLOR, -1, cv2.LINE_AA)
        # Segment label at midpoint (units only)
        seg_px = distance(p1, p2)
        mid_world = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        mid_screen = world_to_screen(mid_world)
        label = format_length_only_units(seg_px)
        if AUTO_CALIBRATE_FROM_FIRST and seg_idx == 1 and pixel_per_unit and unit_label:
            label = f"{KNOWN_FIRST_LENGTH:.4f} {unit_label} (cal)"
        cv2.putText(canvas, label, (int(mid_screen[0]) + 8, int(mid_screen[1]) - 8),
                    FONT, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    # Draw preview from first_point to current mouse (if any)
    if draw_hover_preview is not None and first_point is not None:
        s1 = world_to_screen(first_point)
        s2 = draw_hover_preview
        cv2.line(canvas, s1, s2, (200, 200, 80), THICKNESS, cv2.LINE_AA)
        cv2.circle(canvas, s1, max(2, int(POINT_RADIUS)), (200, 200, 80), -1, cv2.LINE_AA)

def draw_overlay_world(img):
    """Draw overlays directly on a copy of the full-res image (world space)."""
    vis = img.copy()

    for seg_idx, (p1, p2) in enumerate(segments, start=1):
        cv2.line(vis, p1, p2, LINE_COLOR, THICKNESS, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(vis, p, POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)
        seg_px = distance(p1, p2)
        mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        label = format_length_only_units(seg_px)
        if AUTO_CALIBRATE_FROM_FIRST and seg_idx == 1 and pixel_per_unit and unit_label:
            label = f"{KNOWN_FIRST_LENGTH:.4f} {unit_label} (cal)"
        cv2.putText(vis, label, (mid[0]+8, mid[1]-8), FONT, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    # Show cumulative (use units if calibrated, else px)
    cum_px = cumulative_length_px()
    if pixel_per_unit and unit_label:
        cum_label = f"Total length: {cum_px / pixel_per_unit:.3f} {unit_label}"
    else:
        cum_label = f"Total length: {cum_px:.2f} px"
    (tw, th), _ = cv2.getTextSize(cum_label, FONT, 0.7, 2)
    cv2.rectangle(vis, (10, 10), (10 + tw + 10, 10 + th + 10), (255, 255, 255), -1)
    cv2.putText(vis, cum_label, (15, 10 + th + 2), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Calibration info line
    cal_text = "Not calibrated"
    if pixel_per_unit and unit_label:
        cal_text = f"Scale: {pixel_per_unit:.6f} px per {unit_label}"
    cv2.putText(vis, cal_text, (10, vis.shape[0]-15), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis

def draw_view(img, win_name, mouse_xy=None):
    """Render the current view and overlays."""
    # Get window size
    h, w = img.shape[:2]
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
    except Exception:
        win_w, win_h = int(w * scale), int(h * scale)
    win_w = max(100, win_w)
    win_h = max(100, win_h)

    # Prepare canvas
    canvas = np.full((win_h, win_w, 3), BG_COLOR, dtype=np.uint8)

    # Affine transform to map image -> screen
    M = np.array([[scale, 0, offset[0]],
                  [0, scale, offset[1]]], dtype=np.float32)
    cv2.warpAffine(img, M, (win_w, win_h), dst=canvas, flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_TRANSPARENT)

    # Optional hover preview target (screen coords)
    hover_screen_pt = None
    if mouse_xy is not None and first_point is not None:
        # If exactly one completed segment exists (we're drawing the 2nd),
        # snap the preview to be parallel and start at the same start point.
        if len(segments) == 1 and first_seg_unit is not None:
            wx, wy = screen_to_world(*mouse_xy)
            snapped_world = snap_to_parallel(first_point, (wx, wy))
            hover_screen_pt = world_to_screen(snapped_world)
        else:
            hover_screen_pt = mouse_xy

    # Draw segments and preview
    draw_segments_on_canvas(canvas, draw_hover_preview=hover_screen_pt)

    # Cumulative label (units if calibrated)
    cum_px = cumulative_length_px()
    if pixel_per_unit and unit_label:
        cum_label = f"Total length: {cum_px / pixel_per_unit:.3f} {unit_label}"
    else:
        cum_label = f"Total length: {cum_px:.2f} px"
    (tw, th), _ = cv2.getTextSize(cum_label, FONT, 0.7, 2)
    cv2.rectangle(canvas, (10, 10), (10 + tw + 12, 10 + th + 12), (255, 255, 255), -1)
    cv2.putText(canvas, cum_label, (16, 10 + th + 4), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Calibration info
    cal_text = "Not calibrated"
    if pixel_per_unit and unit_label:
        cal_text = f"Scale: {pixel_per_unit:.6f} px per {unit_label}"
    cv2.putText(canvas, cal_text, (10, win_h - 15), FONT, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

    # Instructions
    instructions = ("Left-click: 1st/2nd point → new segment "
                    "(2nd starts at 1st line start and is ∥) | Wheel: zoom | "
                    "Right/Middle drag: pan | c: manual calibrate | u: undo | r: reset | "
                    "s: save | +/-: zoom | 0: fit | q/ESC: quit")
    (tw2, th2), _ = cv2.getTextSize(instructions, FONT, 0.48, 2)
    cv2.rectangle(canvas, (10, 90), (10 + min(tw2 + 12, win_w - 20), 90 + th2 + 12), (0, 0, 0), -1)
    cv2.putText(canvas, instructions[:max(0, min(len(instructions), int((win_w - 40) / 6.8)))],
                (16, 90 + th2 + 4), FONT, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    # If a first_point is set, show its marker
    if first_point is not None:
        s1 = world_to_screen(first_point)
        cv2.circle(canvas, s1, max(2, int(POINT_RADIUS + 2)), (255, 180, 0), -1, cv2.LINE_AA)
        cv2.putText(canvas, "First point selected", (s1[0] + 8, s1[1] - 18), FONT, 0.5, (255, 220, 120), 1, cv2.LINE_AA)

    return canvas

def save_annotated(img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(IMAGE_PATH)
    out_path = f"{root}_measured_{timestamp}{ext if ext else '.png'}"
    cv2.imwrite(out_path, img)
    print(f"[Saved] {out_path}")

# ---------------------------
# Mouse + keyboard handlers
# ---------------------------
def zoom_at_screen_pos(x, y, zoom_in=True):
    global scale, offset
    # Convert screen point to world before zoom
    wx, wy = screen_to_world(x, y)
    old_scale = scale
    if zoom_in:
        scale *= ZOOM_STEP
    else:
        scale /= ZOOM_STEP
    scale = clamp_scale(scale)
    if abs(scale - old_scale) < 1e-9:
        return
    # Adjust offset so (wx, wy) stays under (x, y)
    offset[0] = x - wx * scale
    offset[1] = y - wy * scale

def on_mouse(event, x, y, flags, param):
    global segments, first_point, calibrating, calib_points_screen, pixel_per_unit, unit_label
    global panning, pan_start, offset, last_mouse_xy, first_seg_unit

    last_mouse_xy = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if calibrating:
            calib_points_screen.append((x, y))
            if len(calib_points_screen) == 2:
                # Convert both screen points to world for calibration distance
                w0 = screen_to_world(*calib_points_screen[0])
                w1 = screen_to_world(*calib_points_screen[1])
                px = distance(w0, w1)
                print(f"[Calibration] Pixel distance between selected points: {px:.4f} px")
                # Ask for real-world distance via console
                try:
                    real = float(input("Enter real-world distance between these points (numeric): ").strip())
                except Exception:
                    print("Invalid number. Calibration aborted.")
                    calib_points_screen.clear()
                    calibrating = False
                    return
                unit = input("Enter unit label (e.g., mm, cm, in, m): ").strip() or "units"
                if real <= 0:
                    print("Real-world distance must be > 0. Calibration aborted.")
                else:
                    pixel_per_unit = px / real
                    unit_label = unit
                    print(f"[Calibration] Set: {pixel_per_unit:.6f} px per {unit_label}")
                calib_points_screen.clear()
                calibrating = False
        else:
            wx, wy = screen_to_world(x, y)
            clicked_world = (float(wx), float(wy))

            if first_point is None:
                # Starting a new segment
                if len(segments) == 1:
                    # Lock the second line's start to the FIRST endpoint of the first line.
                    # If you prefer the second line to start from the first line's END instead, change [0] -> [1].
                    locked_start = segments[0][0]  # <- change to segments[0][1] to use the other endpoint
                    first_point = (int(locked_start[0]), int(locked_start[1]))
                    print("[Info] Second line start locked to the first line's start point.")
                else:
                    first_point = (int(round(clicked_world[0])), int(round(clicked_world[1])))
            else:
                # Completing a segment
                end_world = clicked_world

                # If we're creating the SECOND segment, snap the endpoint to be parallel to the first
                if len(segments) == 1 and first_seg_unit is not None:
                    end_world = snap_to_parallel(first_point, end_world)

                new_seg = (first_point,
                           (int(round(end_world[0])), int(round(end_world[1]))))
                segments.append(new_seg)

                # If this is the first segment and auto-calibration is enabled, set scale + direction
                if AUTO_CALIBRATE_FROM_FIRST and len(segments) == 1:
                    seg_px = distance(*new_seg)
                    if KNOWN_FIRST_LENGTH > 0:
                        pixel_per_unit = seg_px / KNOWN_FIRST_LENGTH
                        unit_label = KNOWN_UNIT
                        print(f"[Auto-calibration] First segment set to {KNOWN_FIRST_LENGTH} {KNOWN_UNIT}.")
                        print(f"[Auto-calibration] Scale: {pixel_per_unit:.6f} px per {unit_label}")
                    else:
                        print("[Auto-calibration] KNOWN_FIRST_LENGTH must be > 0; skipping calibration.")
                    # Store first direction
                    set_first_direction(new_seg)

                # If this is the second segment, print its length in inches
                if len(segments) == 2 and pixel_per_unit and unit_label:
                    seg2_px = distance(*segments[1])
                    seg2_real = seg2_px / pixel_per_unit
                    print(f"Second segment length: {seg2_real:.3f} {unit_label}")

                first_point = None

    elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        panning = True
        pan_start = np.array([x, y], dtype=float)

    elif event == cv2.EVENT_MOUSEMOVE and panning:
        delta = np.array([x, y], dtype=float) - pan_start
        offset += delta
        pan_start = np.array([x, y], dtype=float)

    elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
        panning = False
        pan_start = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        # Mouse wheel zoom
        try:
            delta = cv2.getMouseWheelDelta(flags)
        except Exception:
            delta = 120 if flags > 0 else -120
        zoom_at_screen_pos(x, y, zoom_in=(delta > 0))

# ---------------------------
# Main
# ---------------------------
def main():
    global segments, first_point, calibrating, calib_points_screen, pixel_per_unit, unit_label
    global scale, offset, last_mouse_xy, first_seg_unit

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Could not load image: {IMAGE_PATH}")
        return

    win = "Image Distance Measure (Auto-Calibrate, ∥ 2nd Segment from same start, Zoom/Pan)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # resizable
    cv2.setMouseCallback(win, on_mouse)

    # Initial fit
    fit_to_window(img.shape, win)

    while True:
        vis = draw_view(img, win, mouse_xy=last_mouse_xy)
        cv2.imshow(win, vis)
        key = cv2.waitKey(16) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('u'):
            # Undo: if a first_point is active, cancel it; else remove last segment
            if first_point is not None:
                first_point = None
            elif segments:
                popped = segments.pop()
                # If we removed the first segment, clear its stored direction
                if len(segments) == 0:
                    first_seg_unit = None
        elif key == ord('r'):
            segments = []
            first_point = None
            first_seg_unit = None
        elif key == ord('c'):
            # Optional: manual calibration
            calibrating = True
            calib_points_screen = []
            print("Manual calibration: Click two points with a known real-world distance...")
        elif key == ord('s'):
            annotated = draw_overlay_world(img)
            save_annotated(annotated)
        elif key in (ord('+'), ord('=')):  # '=' often same key as '+'
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(win)
            except Exception:
                win_w, win_h = vis.shape[1], vis.shape[0]
            zoom_at_screen_pos(win_w // 2, win_h // 2, zoom_in=True)
        elif key in (ord('-'), ord('_')):
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(win)
            except Exception:
                win_w, win_h = vis.shape[1], vis.shape[0]
            zoom_at_screen_pos(win_w // 2, win_h // 2, zoom_in=False)
        elif key == ord('0'):
            fit_to_window(img.shape, win)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
