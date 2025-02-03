'''Calculates z-scores for all observations.
'''
import numpy as np
import pandas as pd

def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def cols_as_np(batch, cols):
    # Convert to numpy array
    ctrls = batch[cols].values
    ctrls[~np.frompyfunc(isnumeric, 1,1)(ctrls).astype(bool)] = np.nan
    ctrls = ctrls.astype(float)
    return ctrls

def z_scores(batch, null_cols, target_cols):
    ctrls = cols_as_np(batch, null_cols)
    targets = cols_as_np(batch, target_cols)
    # Add any large values to the null control collection since we assume these
    # drugs have at-worst no effect.
    updated = np.zeros_like(targets)
    updated[:] = np.nan
    for i,(ctrl, target) in enumerate(zip(ctrls, targets)):
        ctrl_max = np.nanmax(ctrl)
        u = list(ctrl) + list(target[(target > ctrl_max) & (~np.isnan(target))])
        updated[i, :len(u)] = u
    ctrls = updated
    ctrl_mean = np.nanmean(ctrls, axis=1)
    ctrl_std = np.nanstd(ctrls, axis=1)
    return (targets - ctrl_mean[:,np.newaxis]) / ctrl_std[:,np.newaxis], ctrl_mean, ctrl_std

if __name__ == '__main__':
    # Load the dataset (should be the filtered dataset from step2)
    print('Loading data')
    df = pd.read_csv('data/nutlin/all_outcomes.csv', header=0, delimiter=',')

    # Total number of each well type
    npos = 48
    nneg = 32
    ndose = 9

    # Get the names of the negative and positive control columns and the treatment columns
    neg_cols = ['blank{}'.format(i) for i in range(1,nneg+1)]
    pos_cols = ['control{}'.format(c) for c in range(1,npos+1)]
    treatment_cols = ['raw_max'] + ['raw{}'.format(i) for i in range(2,ndose+1)]
    
    # Correct some dosage differences in GDSC
    Y = df[treatment_cols].values
    select = np.any(np.isnan(Y), axis=1)
    Y[select,0::2] = Y[select,:5]
    Y[select,1::2] = np.nan
    df[treatment_cols] = Y

    # Convert everything to Z scores
    print('Converting observations to Z scores')
    z_treatment_cols = ['z_dose{}'.format(i) for i in range(len(treatment_cols))]
    z_pos_cols = ['z_control{}'.format(i) for i in range(len(pos_cols))]
    z_neg_cols = ['z_blank{}'.format(i) for i in range(len(neg_cols))]
    z_cols, ctrl_mean, ctrl_std = z_scores(df, pos_cols, neg_cols+pos_cols+treatment_cols)
    for c,r in zip(z_neg_cols+z_pos_cols+z_treatment_cols, z_cols.T):
        df[c] = pd.Series(r)
    df['control_mean'] = pd.Series(ctrl_mean)
    df['control_std'] = pd.Series(ctrl_std)

    print('Saving')
    df.to_csv('data/nutlin/all_outcomes.csv', index=False)

