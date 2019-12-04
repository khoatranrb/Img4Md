# Project: Xe bluetooth tránh vật cản

*Sinh viên: Trần Văn Khoa*



## I. Mục tiêu:

* Xe robot được điều khiển qua bluetooth
* Tránh vật cản trước mắt

## II. Chuẩn bị linh kiện:

### 1. Board Arduino Uno R3

![Arduino Uno R3](https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/uno.jpg)

* Board Arduino được dùng như trung tâm điều khiển của robot.

### 2. Motor shield L293D

![](https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/l293.jpg)

* Module L293D có vai trò điều khiển motor.

### 3. Motor DC(x2)

<p>
    <img src='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/dc.jpg' width=200>
</p>

* Motor DC là động cơ chính của xe.

### 4. Module bluetooth HC-06

<p>
    <img src='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/nlybl1.png' width=300>
</p>

* Module bluetooth HC-06 phát bluetooth, kết nối board Arduino với thiết bị điều khiển (thiết bị Android).
* Khoảng cách truyền khoảng 15m.

### 5. Cảm biến siêu âm HR-SC04

<p>
    <img src='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/cbsa.jpg' width=200>
</p>

* Cảm biến siêu âm xác định khoảng cách của xe với vật cản phía trước.

#### 5.1. Nguyên lý hoạt động:

<p>
    <img src='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/cbsa2.png' width=300>
</p>

* Cảm biến phát ra một xung ngắn từ chân **Trig**, sau đó bật *HIGH* chân **Echo** đến khi nhận lại tín hiệu đã phát

  <p>
      <img src ='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/nlyhcsr.jpg' width=300>
  </p>

* Từ thời gian phát đến thời gian tín hiệu thu, tính được khoảng cách đến vật cản.

* Khoảng cách đến vật cản:
  $$
  distance = \frac{t}{2}\times \frac{340}{1000}
  $$
  

  * vận tốc âm thanh 340 $m/s$

### 6. Linh kiện khác

* Đế pin, pin, công tắc
* Dây nối
* Khung xe, bánh xe...

## III. Lắp ráp

* Lắp module L293D với Arduino như hình:

  <p>
      <img src ='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/l293Uno.png' width=300>
  </p>

* Thiết lập cảm biến khoảng cách:

  <p>
      <img src ='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/hcsr04.jpg' width=200>
  </p>

* Thiết lập module bluetooth:

  <p>
      <img src ='https://raw.githubusercontent.com/khoatranrb/Img4Md/master/TN%26KP_RB/bl06.jpg' width=400>
  </p>

  

## IV. Giải thuật:

















### Tham khảo:

[1] https://www.slideshare.net/trongthuy2/luan-van-mach-dieu-khien-thiet-bi-bang-android-bang-bluetooth-hay