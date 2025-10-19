import copy

class ClassMapper:
  def __init__(self):
    self.num_classes = 0
    self.label_to_class_id = dict()
    self.class_id_to_name = dict()
    self.class_id_to_label = dict()
    self.name_to_class_id = dict()
  def Register(self, label_id, label_name):
    if label_id in self.label_to_class_id: return
    class_id = self.num_classes
    self.num_classes += 1
    self.label_to_class_id[label_id] = class_id
    self.class_id_to_name[class_id] = label_name
    self.class_id_to_label[class_id] = label_id
    self.name_to_class_id[label_name] = class_id
  def LabelToClass(self, label_id):
    return self.label_to_class_id[label_id]
  def ClassToName(self, class_id):
    return self.class_id_to_name[class_id]
  def ClassToLabel(self, class_id):
    return self.class_id_to_label[class_id]
  def NameToClass(self, name):
    return self.name_to_class_id[name]
  def Copy(self):
    return copy.deepcopy(self)
  def __len__(self):
    return self.num_classes