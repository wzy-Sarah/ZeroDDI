

class Evaluator:
    def __init__(self, dset, model):

        self.dset = dset

        # convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # mask over pairs that occur in closed world
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with val pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
        self.test_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                           for attr, obj in list(test_pair_set)]
        mask = [1 if pair in test_pair_set else 0 for pair in dset.pairs]
        self.closed_mask = torch.ByteTensor(mask)

        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.ByteTensor(mask)

        # object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.ByteTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # decide if the model being evaluated is a manifold model or not
        mname = model.__class__.__name__
        if 'VisualProduct' in mname:
            self.score_model = self.score_clf_model
        else:
            self.score_model = self.score_manifold_model

    # generate masks for each setting, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth):  # (B, #pairs)
        def get_pred_from_scores(_scores):
            _, pair_pred = _scores.topk(10, dim=1)  #sort(1, descending=True)
            pair_pred = pair_pred[:, :10].contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, 10), self.pairs[pair_pred][:, 1].view(-1, 10)
            return (attr_pred, obj_pred)

        results = {}

        # open world setting -- no mask
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[1 - mask] = -1e10
        results.update({'open': get_pred_from_scores(closed_scores)})

        # closed world setting - set the score for all NON test-pairs to -1e10
        #results.update({'closed': get_pred_from_scores(closed_scores)})
        results.update({'closed': results['open']})

        # object_oracle setting - set the score to -1e10 for all pairs where the true object does NOT participate
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[1 - mask] = -1e10
        results.update({
            'object_oracle': get_pred_from_scores(oracle_obj_scores)
        })

        return results

    def score_clf_model(self, scores, obj_truth):

        attr_pred, obj_pred = scores

        # put everything on CPU
        attr_pred, obj_pred, obj_truth = attr_pred.cpu(), obj_pred.cpu(
        ), obj_truth.cpu()

        # - gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # - multiply P(a)*P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (B, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores
        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0):
        # put everything on CPU
        scores = {k: v.cpu() for k, v in scores.items()}
        obj_truth = obj_truth.cpu()
        # gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs],
            1)  # (B, #pairs)
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[1 - mask] += bias
        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores
        results['scores'] = orig_scores
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, topk=1):

        # put everything on cpu
        attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()
        pairs = list(
            zip(list(attr_truth.cpu().numpy()), list(obj_truth.cpu().numpy())))
        seen_ind = torch.LongTensor([
            i for i in range(len(attr_truth)) if pairs[i] in self.train_pairs
        ])
        unseen_ind = torch.LongTensor([
            i for i in range(len(attr_truth))
            if pairs[i] not in self.train_pairs
        ])

        # top 1 pair accuracy
        # open world: attribute, object and pair
        attr_match = (attr_truth.unsqueeze(1).repeat(
            1, topk) == predictions['open'][0][:, :topk])
        obj_match = (obj_truth.unsqueeze(1).repeat(
            1, topk) == predictions['open'][1][:, :topk])
        open_match = (attr_match * obj_match).any(1).float()
        attr_match = attr_match.any(1).float()
        obj_match = obj_match.any(1).float()
        open_seen_match = open_match[seen_ind]
        open_unseen_match = open_match[unseen_ind]

        # closed world, obj_oracle: pair
        closed_match = (attr_truth == predictions['closed'][0][:, 0]).float(
        ) * (obj_truth == predictions['closed'][1][:, 0]).float()

        obj_oracle_match = (
            attr_truth == predictions['object_oracle'][0][:, 0]).float() * (
                obj_truth == predictions['object_oracle'][1][:, 0]).float()

        return attr_match, obj_match, closed_match, open_match, obj_oracle_match, open_seen_match, open_unseen_match
