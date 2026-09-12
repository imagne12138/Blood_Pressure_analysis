"""
LightGBM model for PPG-based BP estimation.
Uses the 169-dim Liu2023 features with separate models for SBP and DBP.
"""
import numpy as np
import lightgbm as lgb


class LightGBMRegressor:
    """
    LightGBM wrapper with separate models for SBP and DBP.
    Supports multi-output via two independent LGBMRegressors.
    """

    def __init__(self, sbp_params=None, dbp_params=None):
        self.sbp_params = sbp_params or self._default_params()
        self.dbp_params = dbp_params or self._default_params()
        self.sbp_model = None
        self.dbp_model = None

    @staticmethod
    def _default_params():
        return {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': -1,
            'random_state': 42,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            sbp_params_update=None, dbp_params_update=None,
            **kwargs):
        """
        Train SBP and DBP models.

        Parameters
        ----------
        X_train : np.ndarray  (N, D)
        y_train : np.ndarray  (N, 2)  columns: [SBP, DBP]
        X_val, y_val : optional validation set
        sbp_params_update, dbp_params_update : optional param overrides
        """
        params = self.sbp_params.copy()
        if sbp_params_update:
            params.update(sbp_params_update)
        if X_val is not None and y_val is not None:
            eval_set_sbp = [(X_val, y_val[:, 0])]
        else:
            eval_set_sbp = None

        self.sbp_model = lgb.LGBMRegressor(**params)
        self.sbp_model.fit(
            X_train, y_train[:, 0],
            eval_set=eval_set_sbp,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)] if eval_set_sbp else None,
            **kwargs
        )

        params = self.dbp_params.copy()
        if dbp_params_update:
            params.update(dbp_params_update)
        if X_val is not None and y_val is not None:
            eval_set_dbp = [(X_val, y_val[:, 1])]
        else:
            eval_set_dbp = None

        self.dbp_model = lgb.LGBMRegressor(**params)
        self.dbp_model.fit(
            X_train, y_train[:, 1],
            eval_set=eval_set_dbp,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)] if eval_set_dbp else None,
            **kwargs
        )

    def predict(self, X):
        """Return (N, 2) array: [SBP_pred, DBP_pred]."""
        sbp_pred = self.sbp_model.predict(X)
        dbp_pred = self.dbp_model.predict(X)
        return np.column_stack([sbp_pred, dbp_pred])

    def feature_importance(self, feature_names=None):
        """Return feature importance dict for both SBP and DBP models."""
        imp = {}
        imp['sbp'] = dict(zip(
            feature_names or range(len(self.sbp_model.feature_importances_)),
            self.sbp_model.feature_importances_
        ))
        imp['dbp'] = dict(zip(
            feature_names or range(len(self.dbp_model.feature_importances_)),
            self.dbp_model.feature_importances_
        ))
        return imp

    def save(self, path):
        """Save both models via joblib."""
        import joblib
        joblib.dump({
            'sbp_model': self.sbp_model,
            'dbp_model': self.dbp_model,
            'sbp_params': self.sbp_params,
            'dbp_params': self.dbp_params,
        }, path)

    @classmethod
    def load(cls, path):
        """Load saved model."""
        import joblib
        data = joblib.load(path)
        instance = cls(
            sbp_params=data.get('sbp_params'),
            dbp_params=data.get('dbp_params')
        )
        instance.sbp_model = data['sbp_model']
        instance.dbp_model = data['dbp_model']
        return instance
