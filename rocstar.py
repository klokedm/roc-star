import torch


def epoch_update_gamma(y_true,y_pred, epoch=-1,delta=1):
        """
        Calculate gamma from last epoch's targets and predictions.
        Gamma is updated at the end of each epoch.
        y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        """
        DELTA = delta  # Fixed: was delta+1, should be delta per Yan et al. 2003 (proportional margin parameter)
        SUB_SAMPLE_SIZE = 2000.0
        # Fixed: Use >=0.5 threshold (consistent with roc_star_loss) to handle soft labels (BIO-R3).
        # Soft labels are any y_true values not exactly 0.0 or 1.0 (e.g., 0.7 for uncertain annotations).
        pos = y_pred[y_true >= 0.5]
        neg = y_pred[y_true < 0.5]
        # subsample the training set for performance
        cap_pos = pos.shape[0]
        cap_neg = neg.shape[0]
        # Fixed: Guard against division by zero (P0 bug)
        if cap_pos == 0 or cap_neg == 0:
            device = y_pred.device if hasattr(y_pred, 'device') else 'cpu'
            return torch.tensor(0.2, dtype=torch.float, device=device)
        pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE/cap_pos]
        neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE/cap_neg]
        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]
        pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
        neg_expand = neg.repeat(ln_pos)
        diff = neg_expand - pos_expand
        ln_All = diff.shape[0]
        Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
        ln_Lp = Lp.shape[0]-1
        diff_neg = -1.0 * diff[diff<0]
        diff_neg = diff_neg.sort()[0]
        ln_neg = diff_neg.shape[0]-1
        ln_neg = max([ln_neg, 0])
        left_wing = int(ln_Lp*DELTA)
        left_wing = max([0,left_wing])
        left_wing = min([ln_neg,left_wing])
        # Fixed: Device-agnostic tensor creation (P1 bug - was .cuda())
        device = y_pred.device if hasattr(y_pred, 'device') else 'cpu'
        default_gamma=torch.tensor(0.2, dtype=torch.float, device=device)
        # Fixed: Guard against empty tensor indexing (P0 bug)
        if diff_neg.shape[0] > left_wing:
           gamma = diff_neg[left_wing]
        else:
           gamma = default_gamma
        L1 = diff[diff>-1.0*gamma]
        ln_L1 = L1.shape[0]
        if epoch > -1 :
            return gamma
        else :
            return default_gamma



def roc_star_loss( _y_true, y_pred, gamma, _epoch_true, epoch_pred):
        """
        Nearly direct loss function for AUC.
        See article,
        C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
        https://github.com/iridiumblue/articles/blob/master/roc_star.md
            _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
            y_pred: `Tensor` . Predictions.
            gamma  : `Float` Gamma, as derived from last epoch.
            _epoch_true: `Tensor`.  Targets (labels) from last epoch.
            epoch_pred : `Tensor`.  Predicions from last epoch.
        """
        #convert labels to boolean
        y_true = (_y_true>=0.50)
        epoch_true = (_epoch_true>=0.50)

        # if batch is either all true or false return zero loss.
        # Fixed: Return true zero instead of tiny random value (P1 bug)
        if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]:
            return torch.tensor(0.0, dtype=torch.float, device=y_pred.device)

        pos = y_pred[y_true]
        neg = y_pred[~y_true]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000 # Max number of positive training samples
        max_neg = 1000 # Max number of negative training samples (fixed comment)
        cap_pos = epoch_pos.shape[0]
        cap_neg = epoch_neg.shape[0]
        # Fixed: Guard against division by zero and use correct divisor (P0 bugs)
        if cap_pos > 0:
            epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
        if cap_neg > 0:
            epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_neg]  # Fixed: was cap_pos

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements against (subsampled) negative elements
        if ln_pos>0 and epoch_neg.shape[0]>0:  # Fixed: Check both conditions
            pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand - pos_expand + gamma
            l2 = diff2[diff2>0]
            m2 = l2 * l2
            len2 = l2.shape[0]
        else:
            # Fixed: Device-agnostic tensor creation (P1 bug - was .cuda())
            m2 = torch.tensor([0], dtype=torch.float, device=y_pred.device)
            len2 = 0

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 and epoch_pos.shape[0]>0:  # Fixed: Check both conditions
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + gamma
            l3 = diff3[diff3>0]
            m3 = l3*l3
            len3 = l3.shape[0]
        else:
            # Fixed: Device-agnostic tensor creation (P1 bug - was .cuda())
            m3 = torch.tensor([0], dtype=torch.float, device=y_pred.device)
            len3=0

        # Fixed: Normalize by actual pair counts, not constants (P1 algorithm bug)
        if len2 > 0 or len3 > 0:
           res2 = (torch.sum(m2)/len2 if len2 > 0 else 0.0) + (torch.sum(m3)/len3 if len3 > 0 else 0.0)
        else:
           res2 = torch.tensor(0.0, dtype=torch.float, device=y_pred.device)

        # Fixed: Also check for INF, not just NaN (P1 bug)
        res2 = torch.where(torch.isnan(res2) | torch.isinf(res2), torch.zeros_like(res2), res2)

        return res2
