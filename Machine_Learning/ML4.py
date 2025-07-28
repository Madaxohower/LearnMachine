import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import os
import copy
import cv2
import albumentations as A

class YOLOAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotation Tool")

        # Paths
        self.image_dir = "dataset/images"
        self.label_dir = "dataset/labels"
        self.classes_file = "classes.txt"

        # Canvas and Frame
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.X)
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Buttons
        self.mode_var = tk.StringVar(value="box")
        self.toggle_button = tk.Button(self.frame, text="Mode: Box", command=self.toggle_mode)
        self.toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_in_button = tk.Button(self.frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_out_button = tk.Button(self.frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.undo_button = tk.Button(self.frame, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.redo_button = tk.Button(self.frame, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.LEFT, padx=5, pady=5)

        # State
        self.image_path = None
        self.tk_img = None
        self.pil_img = None
        self.original_img = None
        self.image_width = 0
        self.image_height = 0
        self.display_width = 0
        self.display_height = 0
        self.start_x = self.start_y = 0
        self.current_points = []
        self.rect_id = None
        self.poly_id = None
        self.boxes = []  # (class_id, x_center, y_center, width, height)
        self.polys = []  # (class_id, [(x1, y1), (x2, y2), ...])
        self.rect_ids = []
        self.poly_ids = []
        self.text_ids = []
        self.point_ids = []  # For anchor points
        self.class_list = []
        self.base_scale = 1.0
        self.zoom_factor = 1.0
        self.annotation_mode = "box"
        self.annotating = False
        self.editing_poly_index = None
        self.selected_point_index = None
        self.dragging_point = False
        self.undo_stack = []  # Store (boxes, polys) states
        self.redo_stack = []  # Store redo states

        if os.path.exists(self.classes_file):
            with open(self.classes_file) as f:
                self.class_list = [line.strip() for line in f.readlines()]

        # Menu
        menu = tk.Menu(root)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_command(label="Augment Image", command=self.augment_image)
        menu.add_cascade(label="File", menu=file_menu)
        root.config(menu=menu)

        # Bindings
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Button-3>", self.close_polygon)
        self.canvas.bind("<Double-Button-1>", self.edit_polygon)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Delete>", self.delete_selected_point)

    def save_state(self):
        """Save current annotation state to undo stack and clear redo stack."""
        self.undo_stack.append((copy.deepcopy(self.boxes), copy.deepcopy(self.polys)))
        self.redo_stack.clear()

    def undo(self):
        """Revert to previous annotation state."""
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        self.redo_stack.append((copy.deepcopy(self.boxes), copy.deepcopy(self.polys)))
        self.boxes, self.polys = self.undo_stack.pop()
        self.editing_poly_index = None
        self.selected_point_index = None
        self.dragging_point = False
        self.current_points = []
        if self.poly_id:
            self.canvas.delete(self.poly_id)
            self.poly_id = None
        self.redraw_annotations()

    def redo(self):
        """Reapply the next annotation state."""
        if not self.redo_stack:
            messagebox.showinfo("Redo", "Nothing to redo.")
            return
        self.undo_stack.append((copy.deepcopy(self.boxes), copy.deepcopy(self.polys)))
        self.boxes, self.polys = self.redo_stack.pop()
        self.editing_poly_index = None
        self.selected_point_index = None
        self.dragging_point = False
        self.current_points = []
        if self.poly_id:
            self.canvas.delete(self.poly_id)
            self.poly_id = None
        self.redraw_annotations()

    def toggle_mode(self):
        if self.annotating and self.annotation_mode == "polygon":
            messagebox.showwarning("Warning", "Finish or cancel current polygon before switching modes.")
            return
        self.annotation_mode = "polygon" if self.annotation_mode == "box" else "box"
        self.toggle_button.config(text=f"Mode: {self.annotation_mode.capitalize()}")
        self.editing_poly_index = None
        self.selected_point_index = None
        self.dragging_point = False
        self.current_points = []
        if self.poly_id:
            self.canvas.delete(self.poly_id)
            self.poly_id = None
        self.redraw_annotations()

    def load_image(self):
        path = filedialog.askopenfilename(initialdir=self.image_dir, filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.image_path = path
        try:
            self.original_img = Image.open(path)
            self.image_width, self.image_height = self.original_img.size
            max_size = 800
            if self.image_width > max_size or self.image_height > max_size:
                self.base_scale = min(max_size / self.image_width, max_size / self.image_height)
            else:
                self.base_scale = 1.0
            self.zoom_factor = 1.0
            self.update_image()
            self.boxes = []
            self.polys = []
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.editing_poly_index = None
            self.selected_point_index = None
            self.current_points = []
            self.redraw_annotations()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def update_image(self):
        total_scale = self.base_scale * self.zoom_factor
        self.display_width = int(self.image_width * total_scale)
        self.display_height = int(self.image_height * total_scale)
        self.pil_img = self.original_img.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.redraw_annotations()

    def zoom(self, event):
        if not self.image_path:
            return
        if event.delta > 0 or event.num == 4:
            self.zoom_factor *= 1.1
        elif event.delta < 0 or event.num == 5:
            self.zoom_factor /= 1.1
            self.zoom_factor = max(0.1, self.zoom_factor)
        self.update_image()

    def zoom_in(self):
        if not self.image_path:
            return
        self.zoom_factor *= 1.1
        self.update_image()

    def zoom_out(self):
        if not self.image_path:
            return
        self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, self.zoom_factor)
        self.update_image()

    def on_resize(self, event):
        return  # Skip resizing dynamically to keep annotations accurate

    def redraw_annotations(self):
        for rect_id in self.rect_ids:
            self.canvas.delete(rect_id)
        for poly_id in self.poly_ids:
            self.canvas.delete(poly_id)
        for text_id in self.text_ids:
            self.canvas.delete(text_id)
        for point_id in self.point_ids:
            self.canvas.delete(point_id[0])
        self.rect_ids.clear()
        self.poly_ids.clear()
        self.text_ids.clear()
        self.point_ids.clear()

        total_scale = self.base_scale * self.zoom_factor
        for box in self.boxes:
            class_id, x_center, y_center, width, height = box
            x1 = (x_center - width / 2) * self.display_width
            y1 = (y_center - height / 2) * self.display_height
            x2 = (x_center + width / 2) * self.display_width
            y2 = (y_center + height / 2) * self.display_height
            rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red")
            text = self.class_list[class_id]
            text_id = self.canvas.create_text(x1, y1 - 10, text=text, anchor="nw", fill="white")
            self.rect_ids.append(rect_id)
            self.text_ids.append(text_id)

        for i, poly in enumerate(self.polys):
            class_id, points = poly
            scaled_points = [(x * self.display_width, y * self.display_height) for x, y in points]
            poly_id = self.canvas.create_polygon(scaled_points, outline="blue", fill="", stipple="gray25")
            text_x = scaled_points[0][0]
            text_y = scaled_points[0][1] - 10
            text = self.class_list[class_id]
            text_id = self.canvas.create_text(text_x, text_y, text=text, anchor="nw", fill="white")
            self.poly_ids.append(poly_id)
            self.text_ids.append(text_id)

            if i == self.editing_poly_index:
                for j, (x, y) in enumerate(scaled_points):
                    point_id = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="yellow", outline="black")
                    self.point_ids.append((point_id, i, j))

        if self.current_points and self.annotation_mode == "polygon":
            scaled_points = [(x * self.display_width, y * self.display_height) for x, y in self.current_points]
            if self.poly_id:
                self.canvas.delete(self.poly_id)
            self.poly_id = self.canvas.create_polygon(scaled_points, outline="blue", fill="", stipple="gray25")
            for x, y in scaled_points:
                point_id = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="yellow", outline="black")
                self.point_ids.append((point_id, None, len(self.point_ids)))

    def mouse_down(self, event):
        if self.annotation_mode == "box":
            self.start_x, self.start_y = event.x, event.y
            self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")
            self.annotating = True
        else:
            if self.editing_poly_index is not None:
                for point_id, poly_idx, point_idx in self.point_ids:
                    x, y = self.canvas.coords(point_id)[0] + 5, self.canvas.coords(point_id)[1] + 5
                    if abs(event.x - x) < 10 and abs(event.y - y) < 10:
                        self.selected_point_index = point_idx
                        self.dragging_point = True
                        return
                self.dragging_point = False
            else:
                self.current_points.append((event.x / self.display_width, event.y / self.display_height))
                self.redraw_annotations()

    def mouse_drag(self, event):
        if self.annotation_mode == "box":
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
        elif self.annotation_mode == "polygon" and self.editing_poly_index is not None and self.dragging_point:
            if self.selected_point_index is not None:
                self.save_state()
                x, y = event.x / self.display_width, event.y / self.display_height
                self.polys[self.editing_poly_index][1][self.selected_point_index] = (x, y)
                self.redraw_annotations()

    def mouse_up(self, event):
        if not self.annotating:
            return
        if self.annotation_mode == "box":
            x1, y1 = self.start_x, self.start_y
            x2, y2 = event.x, event.y
            label = simpledialog.askstring("Class", "Enter class label:")
            if not label:
                self.canvas.delete(self.rect_id)
                self.annotating = False
                return
            if label not in self.class_list:
                self.class_list.append(label)
                with open(self.classes_file, "a") as f:
                    f.write(label + "\n")

            self.save_state()
            class_id = self.class_list.index(label)
            x1, y1 = min(x1, x2), min(y1, y2)
            x2, y2 = max(x1, x2), max(y1, y2)

            x_center = ((x1 + x2) / 2) / self.display_width
            y_center = ((y1 + y2) / 2) / self.display_height
            width = (x2 - x1) / self.display_width
            height = (y2 - y1) / self.display_height

            self.boxes.append((class_id, x_center, y_center, width, height))
            self.redraw_annotations()
            self.annotating = False
        elif self.annotation_mode == "polygon" and self.dragging_point:
            self.dragging_point = False
            self.selected_point_index = None
            self.redraw_annotations()

    def close_polygon(self, event):
        if self.annotation_mode != "polygon" or len(self.current_points) < 3:
            return
        label = simpledialog.askstring("Class", "Enter class label:")
        if not label:
            if self.poly_id:
                self.canvas.delete(self.poly_id)
            self.current_points = []
            self.poly_id = None
            self.annotating = False
            return
        if label not in self.class_list:
            self.class_list.append(label)
            with open(self.classes_file, "a") as f:
                f.write(label + "\n")

        self.save_state()
        class_id = self.class_list.index(label)
        self.polys.append((class_id, self.current_points[:]))
        self.current_points = []
        self.poly_id = None
        self.annotating = False
        self.redraw_annotations()

    def edit_polygon(self, event):
        if self.annotation_mode != "polygon" or self.annotating:
            return
        for i, poly in enumerate(self.polys):
            class_id, points = poly
            scaled_points = [(x * self.display_width, y * self.display_height) for x, y in points]
            x_coords = [x for x, y in scaled_points]
            y_coords = [y for x, y in scaled_points]
            if min(x_coords) <= event.x <= max(x_coords) and min(y_coords) <= event.y <= max(y_coords):
                self.editing_poly_index = i
                self.redraw_annotations()
                return
        self.editing_poly_index = None
        self.selected_point_index = None
        self.redraw_annotations()

    def delete_selected_point(self, event):
        if self.annotation_mode != "polygon" or self.editing_poly_index is None or self.selected_point_index is None:
            return
        if len(self.polys[self.editing_poly_index][1]) <= 3:
            messagebox.showwarning("Warning", "Polygon must have at least 3 points.")
            return
        self.save_state()
        self.polys[self.editing_poly_index][1].pop(self.selected_point_index)
        self.selected_point_index = None
        self.redraw_annotations()

    def save_annotations(self):
        if not self.image_path or (not self.boxes and not self.polys):
            messagebox.showwarning("No Data", "No annotations to save.")
            print("No annotations: boxes=", self.boxes, "polys=", self.polys)
            return

        relative_image_path = os.path.relpath(self.image_path, self.image_dir)
        subset_folder = relative_image_path.split(os.sep)[0]
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        label_subdir = os.path.join(self.label_dir, subset_folder)
        print(f"Saving to directory: {label_subdir}, file: {base}.txt")

        try:
            os.makedirs(label_subdir, exist_ok=True)
            label_path = os.path.join(label_subdir, base + ".txt")

            print(f"Boxes to save: {self.boxes}")
            print(f"Polygons to save: {self.polys}")

            with open(label_path, "w") as f:
                for box in self.boxes:
                    f.write(" ".join(map(str, box)) + "\n")
                for poly in self.polys:
                    class_id, points = poly
                    coords = [coord for point in points for coord in point]
                    f.write(f"{class_id} {' '.join(map(str, coords))}\n")

            messagebox.showinfo("Saved", f"Annotations saved to {label_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
            print(f"Error saving: {str(e)}")

    def augment_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Load an image first.")
            return

        if not self.boxes and not self.polys:
            messagebox.showwarning("No Annotations", "Please annotate before augmenting.")
            return

        image = cv2.imread(self.image_path)
        h, w, _ = image.shape
        bboxes = []
        class_ids_boxes = []
        for box in self.boxes:
            class_id, x, y, bw, bh = box
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            bboxes.append([x1, y1, x2, y2])
            class_ids_boxes.append(class_id)

        polys_for_aug = []
        class_ids_polys = []
        for poly in self.polys:
            class_id, points = poly
            x_coords = [p[0] * w for p in points]
            y_coords = [p[1] * h for p in points]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            bboxes.append([x1, y1, x2, y2])
            class_ids_boxes.append(class_id)
            polys_for_aug.append((class_id, points))
            class_ids_polys.append(class_id)

        transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        relative_image_path = os.path.relpath(self.image_path, self.image_dir)
        subset_folder = relative_image_path.split(os.sep)[0]
        subset_image_dir = os.path.join(self.image_dir, subset_folder)
        subset_label_dir = os.path.join(self.label_dir, subset_folder)

        os.makedirs(subset_image_dir, exist_ok=True)
        os.makedirs(subset_label_dir, exist_ok=True)

        for i in range(3):
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_ids_boxes)
            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_filename = f"aug_{i}_" + os.path.basename(self.image_path)
            aug_path = os.path.join(subset_image_dir, aug_filename)
            cv2.imwrite(aug_path, aug_img)

            h, w, _ = aug_img.shape
            base = os.path.splitext(aug_filename)[0]
            label_path = os.path.join(subset_label_dir, base + ".txt")
            with open(label_path, "w") as f:
                for bbox, class_id in zip(aug_boxes[:len(self.boxes)], class_ids_boxes[:len(self.boxes)]):
                    x1, y1, x2, y2 = bbox
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h
                    f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
                for class_id, points in polys_for_aug:
                    coords = [coord for point in points for coord in point]
                    f.write(f"{class_id} {' '.join(map(str, coords))}\n")

        messagebox.showinfo("Augmented", f"Saved 3 augmented images and labels to their respective folders.")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnnotationTool(root)
    root.mainloop()