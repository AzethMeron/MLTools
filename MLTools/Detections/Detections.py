import math
import numpy as np

# Used for loading, plotting and manipulating detection boxes.
# Uses CXCYWH
class Detection:
    def __init__(self, cx, cy, w, h, rotation, class_id, confidence):
      self.cx = cx
      self.cy = cy
      self.w = w
      self.h = h
      self.rotation = rotation
      self.class_id = class_id
      self.confidence = confidence
    def Center(self):
      return self.cx, self.cy
    def Width(self):
      return self.w
    def Height(self):
      return self.h
    def ClassId(self):
      return self.class_id
    def Confidence(self):
      return self.confidence
    def Rotation(self):
      return self.rotation
    def __str__(self):
      return f"Detection(x={self.cx}, y={self.cy}, w={self.w}, h={self.h}, angle={self.rotation}, class_id={self.class_id}, confidence={self.confidence})"
    def __repr__(self):
      return str(self)
    @staticmethod
    def from_xyxy(x1, y1, x2, y2, class_id, confidence):
      w = x2 - x1
      h = y2 - y1
      x = x1 + w / 2
      y = y1 + h / 2
      return Detection(x, y, w, h, 0, class_id, confidence)
    @staticmethod
    def from_topleft_xywh(x, y, w, h, class_id, confidence):
      x = x + w / 2
      y = y + h / 2
      return Detection(x, y, w, h, 0, class_id, confidence)
    @staticmethod
    def from_xywh(x, y, w, h, class_id, confidence):
      return Detection.from_topleft_xywh(x,y,w,h,class_id,confidence)
    @staticmethod
    def from_cxcywh(cx, cy, w, h, class_id, confidence):
      return Detection(cx, cy, w, h, 0, class_id, confidence)
    def Explode(self):
      return (self.cx, self.cy, self.w, self.h, self.rotation), self.class_id, self.confidence
    def Rescale(self, orig_size, new_size):
      orig_W, orig_H = orig_size
      new_W, new_H = new_size
      cx = self.cx * (new_W / orig_W)
      cy = self.cy * (new_H / orig_H)
      w = self.w * (new_W / orig_W)
      h = self.h * (new_H / orig_H)
      return Detection(cx, cy, w, h, self.rotation, self.class_id, self.confidence)
    def Rotate(self, angle, rotation_center):
          """Rotate the detection box around a given center by `angle` degrees."""
          angle_rad = math.radians(angle)
          cx, cy = rotation_center

          dx = self.cx - cx
          dy = self.cy - cy

          cos_a = math.cos(angle_rad)
          sin_a = math.sin(angle_rad)

          new_x = cx + cos_a * dx - sin_a * dy
          new_y = cy + sin_a * dx + cos_a * dy

          new_rotation = (self.rotation + angle) % 360
          return Detection(new_x, new_y, self.w, self.h, new_rotation, self.class_id, self.confidence)

    def RotatedCorners(self):
          """Return list of four (x, y) corner points after applying rotation."""
          angle_rad = math.radians(self.rotation)
          cos_a = math.cos(angle_rad)
          sin_a = math.sin(angle_rad)

          hw = self.w / 2
          hh = self.h / 2

          # Define corners relative to center
          local_corners = [
              (-hw, -hh),
              ( hw, -hh),
              ( hw,  hh),
              (-hw,  hh)
          ]

          # Apply rotation and translate to global coords
          return [
              (
                  self.cx + dx * cos_a - dy * sin_a,
                  self.cy + dx * sin_a + dy * cos_a
              )
              for dx, dy in local_corners
          ]
    def HorizontalFlip(self, image_width):
      flipped_x = image_width - self.cx
      flipped_rotation = (-self.rotation) % 360
      return Detection(flipped_x, self.cy, self.w, self.h, flipped_rotation, self.class_id, self.confidence)

    def VerticalFlip(self, image_height):
      flipped_y = image_height - self.cy
      flipped_rotation = (180 - self.rotation) % 360
      return Detection(self.cx, flipped_y, self.w, self.h, flipped_rotation, self.class_id, self.confidence)

    def TranslateW(self, dx):
      return Detection(self.cx + dx, self.cy, self.w, self.h, self.rotation, self.class_id, self.confidence)

    def TranslateH(self, dy):
      return Detection(self.cx, self.cy + dy, self.w, self.h, self.rotation, self.class_id, self.confidence)

    def Draw(self, pil_image, class_id_to_name = None, width=2):
      from PIL import ImageDraw
      draw = ImageDraw.Draw(pil_image)
      corners = self.RotatedCorners()
      draw.polygon(corners, outline="red", width=width)
      if class_id_to_name:
        draw.text((self.cx, self.cy), f"{class_id_to_name(self.class_id)}", fill="red")
      else:
        draw.text((self.cx, self.cy), f"{self.class_id}", fill="red")
      return pil_image

    def ToXYXY(self):
      # Convert to axis-aligned bounding box in xyxy format
      corners = self.RotatedCorners()
      xs = [p[0] for p in corners]
      ys = [p[1] for p in corners]
      return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def ToSupervision(detections):
      import supervision as sv
      if type(detections) == list and type(detections[0]) == list: return [ det.ToSupervision() for det in detections ]
      return sv.Detections(
          xyxy=np.array([det.ToXYXY() for det in detections]),
          confidence=np.array([det.confidence for det in detections]),
          class_id=np.array([det.class_id for det in detections])
          )

    @staticmethod
    def from_supervision(sv_detections):
      if type(sv_detections) == list: return [ Detection.from_supervision(det) for det in sv_detections ]
      import supervision as sv
      output = []
      for bbox, conf, class_id in zip(sv_detections.xyxy, sv_detections.confidence, sv_detections.class_id):
          # bbox is a numpy array: [x1, y1, x2, y2]
          x1, y1, x2, y2 = bbox
          output.append( Detection.from_xyxy(x1, y1, x2, y2, class_id, conf) )
      return output

    @staticmethod
    def NaiveNMS(detections, iou_threshold=0.5): # Ignores classes
      import torch
      from torchvision.ops import nms
      if not detections:
        return []
      boxes = torch.tensor([det.ToXYXY() for det in detections], dtype=torch.float32)
      scores = torch.tensor([det.confidence for det in detections], dtype=torch.float32)
      keep_indices = nms(boxes, scores, iou_threshold).tolist()
      return [detections[i] for i in keep_indices]

    @staticmethod
    def NMS(detections, iou_threshold=0.5): # Performs NMS on every class separately
      classful_nms = dict()
      for det in detections:
        if det.class_id not in classful_nms: classful_nms[det.class_id] = []
        classful_nms[det.class_id].append(det)
      output = []
      for class_id in classful_nms:
        output.extend( Detection.NaiveNMS(classful_nms[class_id], iou_threshold) )
      return output

    def IOU(self, other):
        import torch
        from torchvision.ops import box_iou
        box1 = torch.tensor([self.ToXYXY()])
        box2 = torch.tensor([other.ToXYXY()])
        iou = box_iou(box1, box2).item()
        return iou
