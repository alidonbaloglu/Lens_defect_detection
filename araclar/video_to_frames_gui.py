import os
import sys
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2


class VideoToFramesGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video to Frames")

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.fps_var = tk.StringVar(value="5.0")
        self.prefix_var = tk.StringVar(value="frame")
        self.start_index_var = tk.StringVar(value="1")
        self.min_pad_var = tk.StringVar(value="3")
        self.png_comp_var = tk.StringVar(value="0")

        self.is_running = False
        self.thread = None

        self._build_ui()

    def _build_ui(self) -> None:
        frm = tk.Frame(self.root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Video
        tk.Label(frm, text="Video:").grid(row=0, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.video_path, width=60).grid(row=0, column=1, sticky="we", padx=5)
        tk.Button(frm, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=5)

        # Output
        tk.Label(frm, text="Output Dir:").grid(row=1, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.output_dir, width=60).grid(row=1, column=1, sticky="we", padx=5)
        tk.Button(frm, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5)

        # Settings
        tk.Label(frm, text="Desired FPS:").grid(row=2, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.fps_var, width=10).grid(row=2, column=1, sticky="w", padx=5)

        tk.Label(frm, text="Prefix:").grid(row=3, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.prefix_var, width=10).grid(row=3, column=1, sticky="w", padx=5)

        tk.Label(frm, text="Start Index:").grid(row=4, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.start_index_var, width=10).grid(row=4, column=1, sticky="w", padx=5)

        tk.Label(frm, text="Min Pad Width:").grid(row=5, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.min_pad_var, width=10).grid(row=5, column=1, sticky="w", padx=5)

        tk.Label(frm, text="PNG Compression (0-9):").grid(row=6, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.png_comp_var, width=10).grid(row=6, column=1, sticky="w", padx=5)

        # Progress
        self.progress_var = tk.StringVar(value="Idle")
        tk.Label(frm, textvariable=self.progress_var, anchor="w").grid(row=7, column=0, columnspan=3, sticky="we", pady=(10, 5))

        # Actions
        btn_frame = tk.Frame(frm)
        btn_frame.grid(row=8, column=0, columnspan=3, sticky="we", pady=5)
        tk.Button(btn_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=5)

        frm.columnconfigure(1, weight=1)

    def browse_video(self) -> None:
        path = filedialog.askopenfilename(title="Select video", filetypes=[
            ("Video", ".mp4 .avi .mov .mkv .m4v .wmv .flv"),
            ("All files", "*.*")
        ])
        if path:
            self.video_path.set(path)
            # Suggest output dir next to video
            base = os.path.splitext(os.path.basename(path))[0]
            sug = os.path.join(os.path.dirname(path), f"{base}_frames_mask")
            if not self.output_dir.get():
                self.output_dir.set(sug)

    def browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def start(self) -> None:
        if self.is_running:
            return
        try:
            desired_fps = float(self.fps_var.get())
            if desired_fps <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Desired FPS must be a positive number.")
            return

        video = self.video_path.get().strip()
        out_dir = self.output_dir.get().strip()
        prefix = self.prefix_var.get().strip() or "frame"
        try:
            start_index = int(self.start_index_var.get())
            min_pad = int(self.min_pad_var.get())
            png_comp = int(self.png_comp_var.get())
        except Exception:
            messagebox.showerror("Error", "Start/Pad/Compression must be integers.")
            return
        if not (0 <= png_comp <= 9):
            messagebox.showerror("Error", "PNG compression must be in [0, 9].")
            return
        if not os.path.isfile(video):
            messagebox.showerror("Error", "Video file not found.")
            return
        if not out_dir:
            messagebox.showerror("Error", "Please select output directory.")
            return
        os.makedirs(out_dir, exist_ok=True)

        self.is_running = True
        self.progress_var.set("Starting...")
        self.thread = threading.Thread(target=self._run_extract, args=(video, out_dir, desired_fps, prefix, start_index, min_pad, png_comp), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False
        self.progress_var.set("Stopping...")

    def _run_extract(self, video_path: str, out_dir: str, desired_fps: float, prefix: str, start_index: int, min_pad: int, png_comp: int) -> None:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self._set_status("Error: Failed to open video.")
                self.is_running = False
                return

            src_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if src_fps is None or src_fps <= 1e-6:
                src_fps = 0.0
            if total_frames is None or total_frames < 0:
                total_frames = 0

            # Estimate padding width
            if src_fps > 0 and total_frames > 0:
                duration_s = total_frames / src_fps
                expected_saved = int(math.floor(duration_s * desired_fps + 1.0))
            else:
                expected_saved = 999
            pad_width = max(min_pad, len(str(start_index + max(1, expected_saved))))

            next_save_time = 0.0
            save_interval = 1.0 / max(desired_fps, 1e-6)
            saved = 0
            index = start_index
            frame_idx = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while self.is_running:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if src_fps > 0:
                    t = frame_idx / src_fps
                else:
                    t = float(frame_idx)

                if t + 1e-9 >= next_save_time:
                    filename = f"{prefix}{index:0{pad_width}d}.png"
                    out_path = os.path.join(out_dir, filename)
                    ok = cv2.imwrite(out_path, frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_comp])
                    if not ok:
                        self._set_status(f"Warning: Failed to write {filename}")
                    else:
                        saved += 1
                        index += 1
                        next_save_time += save_interval
                        self._set_status(f"Saved: {filename} (total {saved})")

                frame_idx += 1

            cap.release()
            self._set_status(f"Done. Saved {saved} frames to {out_dir}")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self.is_running = False

    def _set_status(self, text: str) -> None:
        def _update():
            self.progress_var.set(text)
        self.root.after(0, _update)


def main() -> None:
    root = tk.Tk()
    root.geometry("640x300")
    app = VideoToFramesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
