import cv2

from text_utils import format_timestamp, wrap_text


def draw_log_panel(frame, entries, panel_width, pinned_alert=None, line_width_chars=32):
    h, w = frame.shape[:2]
    x0 = max(w - panel_width, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 24
    cv2.putText(
        frame,
        "GROUND STATION LOG",
        (x0 + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )
    y += 20

    if pinned_alert is not None:
        alert_lines = wrap_text(
            f"ALERT [{format_timestamp(pinned_alert['time'])}] {pinned_alert['text']}",
            line_width_chars,
        )
        banner_top = y
        banner_height = 14 + len(alert_lines) * 18
        cv2.rectangle(frame, (x0 + 4, banner_top), (w - 4, banner_top + banner_height), (20, 20, 140), -1)
        cv2.rectangle(frame, (x0 + 4, banner_top), (w - 4, banner_top + banner_height), (0, 80, 255), 2)
        ay = banner_top + 18
        for line in alert_lines:
            cv2.putText(frame, line, (x0 + 12, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            ay += 17
        y = banner_top + banner_height + 10

    for entry in entries:
        if entry["kind"] == "alert":
            color = (0, 140, 255)
        elif entry["kind"] == "ai":
            color = (0, 255, 180)
        else:
            color = (255, 255, 255)
        timestamp = format_timestamp(entry["time"])
        for line in wrap_text(f"[{timestamp}] {entry['text']}", line_width_chars):
            if y > h - 10:
                return frame
            cv2.putText(frame, line, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            y += 17
        y += 6
    return frame