
from datetime import date
import pymysql.cursors
from datetime import datetime
import Edit
import time

today = date.today()
print("Today's date:", today)

# datetime object containing current date and time
now = datetime.now()
print("Convert",str(now.microsecond))
now1 = datetime.now()
print("now =", now)
print("Convert",str(now.microsecond))
print("Convert",str(now1.microsecond))
name="van tuuuu"
a = now.microsecond -now1.microsecond
b = str(now);

print(b)
k = time.strptime(b, '%Y-%m-%d %H:%M:%S.%f')
print(k.tm_min)
Edit.insert("van-tu")
# dd/mm/YY H:M:S
# dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
# print("date and time =", dt_string)