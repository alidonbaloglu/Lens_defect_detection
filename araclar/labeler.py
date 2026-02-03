import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw


class PolygonMaskLabeler:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Polygon Mask Labeler")

        # State
        self.image_path = None
        self.output_dir = None
        self.original_image = None  # PIL Image
        self.display_image = None   # PIL Image (scaled)
        self.photo_image = None     # ImageTk
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Annotation state
        self.current_polygon = []  # list of (x_disp, y_disp)
        self.polygons = []         # list of list of (x_disp, y_disp)
        self.point_handles = []    # canvas points for current polygon
        self.segment_handles = []  # canvas lines for current polygon
        self.poly_handles = []     # canvas polygons for finished ones

        # UI
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_open = tk.Button(toolbar, text="Open Image", command=self.open_image)
        btn_open.pack(side=tk.LEFT, padx=4, pady=4)

        btn_out = tk.Button(toolbar, text="Select Output Dir", command=self.select_output_dir)
        btn_out.pack(side=tk.LEFT, padx=4, pady=4)

        btn_finish = tk.Button(toolbar, text="Finish Polygon", command=self.finish_polygon)
        btn_finish.pack(side=tk.LEFT, padx=4, pady=4)

        btn_undo = tk.Button(toolbar, text="Undo Point", command=self.undo_point)
        btn_undo.pack(side=tk.LEFT, padx=4, pady=4)

        btn_clear = tk.Button(toolbar, text="Clear All", command=self.clear_all)
        btn_clear.pack(side=tk.LEFT, padx=4, pady=4)

        btn_save = tk.Button(toolbar, text="Save Mask", command=self.save_mask)
        btn_save.pack(side=tk.LEFT, padx=4, pady=4)

        self.status_var = tk.StringVar(value="Load an image to start.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.root, bg="#222222")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Configure>", self.on_resize)
        self.root.bind("<Escape>", lambda e: self.clear_current())
        self.root.bind("<Control-z>", lambda e: self.undo_point())
        self.root.bind("<Control-s>", lambda e: self.save_mask())

    # ---------- Image loading and display ----------
    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="Select image", filetypes=[
            ("Images", ".png .jpg .jpeg .bmp .tif .tiff"),
            ("PNG", ".png"), ("JPEG", ".jpg .jpeg"), ("Bitmap", ".bmp"), ("TIFF", ".tif .tiff")
        ])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return

        self.image_path = path
        self.original_image = img
        self._fit_image_to_canvas()
        self._reset_annotations()
        self._update_status()

    def select_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select output directory for masks")
        if directory:
            self.output_dir = directory
            self._update_status()

    def on_resize(self, _event=None) -> None:
        if self.original_image is None:
            return
        self._fit_image_to_canvas()
        self._redraw_annotations()

    def _fit_image_to_canvas(self) -> None:
        if self.original_image is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.original_image.size

        scale = min(cw / iw, ch / ih)
        scale = max(scale, 1e-6)
        nw, nh = int(iw * scale), int(ih * scale)
        self.scale = scale
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        self.display_image = self.original_image.resize((nw, nh), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_image, anchor=tk.NW)

    # ---------- Annotation interactions ----------
    def on_left_click(self, event) -> None:
        if self.original_image is None:
            return
        x, y = event.x, event.y
        # Check if click is inside displayed image area
        if x < self.offset_x or y < self.offset_y:
            return
        if x > self.offset_x + self.display_image.width or y > self.offset_y + self.display_image.height:
            return

        self.current_polygon.append((x, y))
        self._draw_current()
        self._update_status()

    def finish_polygon(self) -> None:
        if len(self.current_polygon) < 3:
            messagebox.showinfo("Info", "A polygon needs at least 3 points.")
            return
        # Draw filled polygon overlay (semi-transparent look via stipple)
        handle = self.canvas.create_polygon(
            *self._flatten(self.current_polygon),
            outline="cyan",
            fill="cyan",
            stipple="gray50",
            width=2
        )
        self.poly_handles.append(handle)
        self.polygons.append(list(self.current_polygon))
        self.clear_current()
        self._update_status()

    def undo_point(self) -> None:
        if not self.current_polygon:
            return
        self.current_polygon.pop()
        if self.point_handles:
            self.canvas.delete(self.point_handles.pop())
        if self.segment_handles:
            self.canvas.delete(self.segment_handles.pop())
        self._update_status()

    def clear_current(self) -> None:
        self.current_polygon.clear()
        for h in self.point_handles:
            self.canvas.delete(h)
        for h in self.segment_handles:
            self.canvas.delete(h)
        self.point_handles.clear()
        self.segment_handles.clear()
        self._update_status()

    def clear_all(self) -> None:
        self.clear_current()
        for h in self.poly_handles:
            self.canvas.delete(h)
        self.poly_handles.clear()
        self.polygons.clear()
        self._update_status()

    def _draw_current(self) -> None:
        n = len(self.current_polygon)
        if n == 0:
            return
        x, y = self.current_polygon[-1]
        r = 3
        self.point_handles.append(self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="yellow", outline="black"))
        if n > 1:
            x0, y0 = self.current_polygon[-2]
            self.segment_handles.append(self.canvas.create_line(x0, y0, x, y, fill="yellow", width=2))

    def _redraw_annotations(self) -> None:
        # Redraw background image first
        self._fit_image_to_canvas()
        # Redraw finished polygons
        for poly in self.polygons:
            handle = self.canvas.create_polygon(
                *self._flatten(poly), outline="cyan", fill="cyan", stipple="gray50", width=2
            )
            self.poly_handles.append(handle)
        # Redraw current polygon
        tmp_points = list(self.current_polygon)
        self.clear_current()
        for px, py in tmp_points:
            self.current_polygon.append((px, py))
            self._draw_current()

    # ---------- Saving mask ----------
    def save_mask(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        if not self.polygons and len(self.current_polygon) < 3:
            if not messagebox.askyesno("No polygons", "No closed polygons. Save empty mask?"):
                return
        if self.output_dir is None:
            messagebox.showinfo("Info", "Select output directory first.")
            return

        iw, ih = self.original_image.size
        mask = Image.new("L", (iw, ih), 0)
        draw = ImageDraw.Draw(mask)

        # Convert display coords back to original coords
        all_polys = list(self.polygons)
        if len(self.current_polygon) >= 3:
            all_polys.append(list(self.current_polygon))

        for poly in all_polys:
            orig_pts = [self._to_original(p) for p in poly]
            # Pillow expects sequence of (x, y)
            draw.polygon(orig_pts, fill=255, outline=255)

        os.makedirs(self.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        out_path = os.path.join(self.output_dir, f"{base}_mask.png")
        try:
            mask.save(out_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mask:\n{e}")
            return

        self.status_var.set(f"Saved mask: {out_path}")
        messagebox.showinfo("Saved", f"Mask saved to:\n{out_path}")

    # ---------- Helpers ----------
    def _to_original(self, pt):
        x_disp, y_disp = pt
        x0 = (x_disp - self.offset_x) / max(self.scale, 1e-6)
        y0 = (y_disp - self.offset_y) / max(self.scale, 1e-6)
        # Clamp to image bounds
        x0 = max(0, min(self.original_image.width - 1, x0))
        y0 = max(0, min(self.original_image.height - 1, y0))
        return (float(x0), float(y0))

    def _flatten(self, pts):
        out = []
        for x, y in pts:
            out.extend([x, y])
        return out

    def _reset_annotations(self) -> None:
        self.clear_all()

    def _update_status(self) -> None:
        img = os.path.basename(self.image_path) if self.image_path else "-"
        out = self.output_dir if self.output_dir else "-"
        cur = len(self.current_polygon)
        done = len(self.polygons)
        self.status_var.set(f"Image: {img}  |  Output: {out}  |  Current pts: {cur}  |  Polygons: {done}")


def main() -> None:
    root = tk.Tk()
    root.geometry("1024x768")
    app = PolygonMaskLabeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()
