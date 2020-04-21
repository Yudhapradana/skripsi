from datetime import date, datetime
import time as _time


def get_today():
	return date.today()


def get_today_str():
	return str(get_today())


def get_now():
	return datetime.now()


def sleep(seconds):
	_time.sleep(seconds)


get_time = get_now


get_date = get_today
