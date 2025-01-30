kits21_labels = {"kidney": 1, "ureter": 2, "artery": 3, "vein": 4, "cyst": 5, "tumor": 6}
visceral_labels = {"left_kidney": 9, "right_kidney": 10, "bladder": 13}
# combine L/R kidney labels into single kidney label
visceral_mapping = {"left_kidney": "kidney", "right_kidney": "kidney"}