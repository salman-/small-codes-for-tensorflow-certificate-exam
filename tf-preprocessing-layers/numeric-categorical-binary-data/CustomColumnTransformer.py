class CustomColumnTransformer:

    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, dt, **transform_params):
        return self.func(dt)

    @staticmethod
    def convert_boolean_to_int(X):
        return X * 1

    def get_feature_names_out(self,X):
        return ['Married']
