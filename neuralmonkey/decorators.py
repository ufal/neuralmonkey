from functools import wraps


def tensor(func):
    @wraps(func)
    def decorate(self, *args, **kwargs):
        attribute_name = "_{}_cached_placeholder".format(func.__name__)
        if not hasattr(self, attribute_name):
            value = func(self, *args, **kwargs)
            setattr(self, attribute_name, value)
        return getattr(self, attribute_name)

    return property(decorate)
