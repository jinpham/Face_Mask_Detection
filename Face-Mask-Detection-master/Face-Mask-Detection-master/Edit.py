# Sử dụng module tiện ích của bạn.
import pymysql.cursors
from datetime import date
from datetime import datetime
import time
import myconnutils

def insert(name):
    name = name.replace("-", " ")
    # name="thanh cong"
    connection = myconnutils.getConnection()
    print ("Connect successful!")

    sql = "SELECT Id FROM listtime WHERE Day = %s"
    sql_1 = "INSERT INTO listtime(Day) values(%s)"
    sql_2 = "INSERT INTO listname(Name,Time) values(%s,%s)"
    sql_3 = "SELECT Id FROM listname WHERE Name = %s"
    sql_4 = "SELECT Id FROM listname WHERE Time = %s"
    sql_5 = "INSERT listtime_listname(Id_Time,Id_Name) values(%s,%s)"
    sql_6 = "select listname.Id,listname.Time from listname,listtime_listname,listtime where Name = %s and listname.Id = listtime_listname.Id_Name and listtime.Id = %s order by Id desc limit 0,1"
    try:
        dateTime = datetime.now()
        cursor = connection.cursor()
        # Thực thi sql và truyền 1 tham số.
        cursor.execute(sql, str(date.today()))
        print("cursor.description: ", cursor.description)
        i = 0
        idDay = 0
        for row in cursor:
            i = i + 1
            idDay = row["Id"]
            print("Id: ",row["Id"])
            break
        if i == 0 :
            print("khong co du lieu thang ngu")
            # cursor = connection.cursor()
            cursor.execute(sql_1, date.today())  #thêm ngày vào list time
            # print("ok")
            # cursor = connection.cursor()
            cursor.execute(sql_2, (name, str(dateTime)))# thêm tên và ngày giờ k đeo khẩu trang
            print("ok")
            # lấy id của ngày trong bảng list time
            cursor = connection.cursor()
            cursor.execute(sql, date.today())
            for row in cursor:
                idDay = row["Id"]
                break
            # lấy id của thằng vừa thêm vào dựa trên ngày giờ
            cursor = connection.cursor()
            cursor.execute(sql_4, str(dateTime))
            for row in cursor:
                idName = row["Id"]
                break
            cursor = connection.cursor()
            cursor.execute(sql_5, (idDay, idName))
            connection.commit()
        if i == 1 :
            print("Co du lieu thang ngu")
            print(name)
            print(idDay)
            cursor = connection.cursor()
            cursor.execute(sql_3, name)

            tempId = 0
            for row in cursor:
                tempId = row["Id"]
                print(tempId)
                print("llllllllllll")
                break

            if tempId > 0:
                cursor.execute(sql_6, (name, idDay))
                print("cl ma")
                tempTime = ""
                for row in cursor:
                    tempTime = row["Time"]
                    break
                # if tempTime != '':
                tempTime = time.strptime(tempTime, '%Y-%m-%d %H:%M:%S.%f')
                if dateTime.hour*3600 - dateTime.minute*60 - tempTime.tm_hour * 3600 - tempTime.tm_min * 60 >= 1200:
                    cursor.execute(sql_2, (name, str(dateTime)))  # thêm tên và ngày giờ k đeo khẩu trang
                    print("ok")
                    # lấy id của ngày trong bảng list time
                    cursor = connection.cursor()
                    cursor.execute(sql, date.today())
                    for row in cursor:
                        idDay = row["Id"]
                        break
                    # lấy id của thằng vừa thêm vào dựa trên ngày giờ
                    cursor = connection.cursor()
                    cursor.execute(sql_4, str(dateTime))
                    for row in cursor:
                        idName = row["Id"]
                        break
                    cursor = connection.cursor()
                    cursor.execute(sql_5, (idDay, idName))
                    connection.commit()
                else:
                    print("haha may ngu vcl may qua ngu")
                    pass
            else:
                cursor.execute(sql_1, date.today())  # thêm ngày vào list time
                # print("ok")
                # cursor = connection.cursor()
                cursor.execute(sql_2, (name, str(dateTime)))  # thêm tên và ngày giờ k đeo khẩu trang
                print("ok1")
                # lấy id của ngày trong bảng list time
                cursor = connection.cursor()
                cursor.execute(sql, date.today())
                for row in cursor:
                    idDay = row["Id"]
                    break
                # lấy id của thằng vừa thêm vào dựa trên ngày giờ
                cursor = connection.cursor()
                cursor.execute(sql_4, str(dateTime))
                for row in cursor:
                    idName = row["Id"]
                    break
                cursor = connection.cursor()
                cursor.execute(sql_5, (idDay, idName))
                connection.commit()
        connection.commit()
    finally:
        # Đóng kết nối
        connection.close()
