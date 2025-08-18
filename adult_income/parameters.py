
numerical_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

label_col            = 'income'

positive_value       = '>50K'

sigmas = {
    'age': 3,
    'workclass': 0.05,
    'education': 0.03,
    'educational-num': 1,
    'marital-status': 0.3,
    'occupation': 0.01,
    'relationship': 0.2,
    'race': 0.05,
    'gender': 0.4,
    'capital-gain': 100,
    'capital-loss': 100,
    'hours-per-week': 2,
    'native-country': 0.03
}
