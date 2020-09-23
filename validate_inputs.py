
def validate_inputs(models):
    check_models(models)

def check_models(models):
    if type(models) not in [tuple, list]:
        raise TypeError('`models` argument must a list or tuple')


def check_datetime_index():
    pass

