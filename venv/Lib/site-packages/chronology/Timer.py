from .get_time import get_now, sleep


class Timer:
	def __init__(self, keep_timstamps=False):
		self._timestamps = []
		self._keep = keep_timstamps
		self._start_time = self.get_timestamp(name='timer starts')
		self._previous_timestamp = self._start_time

	def get_timestamp(self, name):
		this_moment = self.now()
		if self._keep: self._timestamps.append((name, this_moment))
		self._previous_timestamp = this_moment
		return this_moment

	def get_previous_timestamp(self):
		return self._previous_timestamp

	def get_elapsed(self):
		this_moment = get_now()
		if self._keep: self._timestamps.append(('timed at', this_moment))
		result = this_moment - self._previous_timestamp
		self._previous_timestamp = this_moment
		return result

	def get_elapsed_seconds(self):
		delta = self.get_elapsed()
		return delta.seconds + delta.microseconds / 1E6

	@property
	def start_time(self):
		return self._start_time

	@staticmethod
	def now():
		return get_now()

	def start(self):
		self._start_time = self.get_timestamp(name='timer restarted')

	@staticmethod
	def sleep_seconds(seconds):
		sleep(seconds)

