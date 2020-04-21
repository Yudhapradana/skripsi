from .get_time import get_now

def get_elapsed(start, end=None):
	end = end or get_now()
	return end - start

def get_elapsed_seconds(start, end=None):
	delta = get_elapsed(start=start, end=end)
	return delta.seconds + delta.microseconds / 1E6