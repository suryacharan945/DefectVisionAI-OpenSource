
import cv2
import numpy as np
from torch.utils.data import Dataset

class MVTecDataset(Dataset):
    """
    Reusable dataset loader for DefectVision AI.
    df must contain: filepath, category, label, defect_type, is_normal, mask_path
    """
    def __init__(self, dataframe, transform=None, return_mask=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_mask = return_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["filepath"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.return_mask and isinstance(row["mask_path"], str) and row["mask_path"] != "":
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        label = 0 if row["is_normal"] else 1  # 0=normal, 1=defective

        sample = {
            "image": image,
            "label": label,
            "category": row["category"],
            "defect_type": row["defect_type"],
            "filepath": row["filepath"],
        }
        if self.return_mask:
            sample["mask"] = mask

        return sample
