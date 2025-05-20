from .acumen_evaluator import AcumenEvaluator
from .acumen_metrics import Metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json


class AcumenClassificationEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args):
        super(AcumenClassificationEvaluator, self).__init__(args, model, evaluator_args)

        from lib import nomenclature
        from lib import device

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[evaluator_args.dataset]
        self.val_dataloader = self.dataset.val_dataloader(args, kind = evaluator_args.target_split)

        self.device = device

    @property
    def display_name(self):
        return 'classification'

    def trainer_evaluate(self, step = None):
        return self.evaluate(save = False)

    @torch.no_grad()
    def evaluate(self, save = True):
        np.set_printoptions(suppress=True)
        y_pred = defaultdict(list)
        y_true = defaultdict(list)

        for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            outputs = self.model(batch)

            for head in self.args.heads:
                if head.kind != 'classification':
                    continue

                preds = outputs[head.name].probas
                preds = preds.detach().cpu().numpy().tolist()

                labels = batch[head.args.label_key]
                labels = labels.detach().cpu().numpy().tolist()

                y_pred[head.name].extend(preds)
                y_true[head.name].extend(labels)

        results = []
        for head in self.args.heads:
            y_pred = np.array(y_pred[head.name])
            y_true = np.array(y_true[head.name])

            y_pred_am = np.argmax(y_pred, axis = 1)

            results += [
                Metric(
                    name = f'{head.name}@accuracy',
                    value = accuracy_score(y_true, y_pred_am),
                    monotonicity = ['instant', 'up'],
                ),
                Metric(
                    name = f'{head.name}@precision',
                    value = precision_score(y_true, y_pred_am, average = self.evaluator_args.average_kind),
                    monotonicity = ['instant', 'up']
                ),
                Metric(
                    name = f'{head.name}@recall',
                    value = recall_score(y_true, y_pred_am, average = self.evaluator_args.average_kind),
                    monotonicity = ['instant', 'up']
                ),
                Metric(
                    name = f'{head.name}@f1',
                    value = f1_score(y_true, y_pred_am, average = self.evaluator_args.average_kind),
                    monotonicity = ['instant', 'up']
                ),
            ]

        # Maybe a result serializer or something?
        if save:
            with open(f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}_results.json", 'wt') as f:
                json.dump(results, f)

        return results
