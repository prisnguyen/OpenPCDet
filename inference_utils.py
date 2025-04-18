import torch
import json

def run_inference(model, data_batch, class_names, return_probs=False, return_embeddings=False, save_path=None):
    model.eval()
    with torch.no_grad():
        if return_embeddings:
            outputs, embeddings = model(data_batch, return_embeddings=True)
        else:
            outputs = model(data_batch)

        pred_dicts, _ = model.post_processing(outputs, data_batch)

        final_results = []
        for pred_dict in pred_dicts:
            boxes = pred_dict['pred_boxes'].cpu().tolist()
            labels = pred_dict['pred_labels'].cpu().tolist()
            scores = pred_dict['pred_scores'].cpu().tolist()

            result_entry = {
                "bboxes": boxes,
                "labels": [class_names[i - 1] for i in labels],
                "scores": scores
            }

            if return_probs and 'pred_class_probs' in pred_dict:
                probs_tensor = pred_dict['pred_class_probs'].cpu()
                probs_list = probs_tensor.tolist()
                prob_details = []

                for i in range(len(boxes)):
                    prob_details.append({
                        "bbox_index": i,
                        "bbox": boxes[i],
                        "predicted_label": class_names[labels[i] - 1],
                        "class_probs": {class_names[j]: probs_list[i][j] for j in range(len(class_names))}
                    })

                result_entry["class_probs"] = prob_details

            if return_embeddings and embeddings is not None:
                result_entry["embeddings"] = embeddings.cpu().tolist()

            final_results.append(result_entry)

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(final_results, f, indent=4)
            print(f"Saved results to {save_path}")

        return final_results