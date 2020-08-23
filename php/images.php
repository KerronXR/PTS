<?php
$conn = mysqli_connect("78.140.191.36","kerronxy_yacov", "]k2oIl?WBRPh", "kerronxy_users");

if($_SERVER['REQUEST_METHOD'] == 'POST')
 {
 $DefaultId = 0;
 
 $ImageData = $_POST['image_path'];
 
 $ImageName = $_POST['image_name'];

 $GetOldIdSQL ="SELECT id FROM ImageToServerTable ORDER BY id ASC";
 
 $Query = mysqli_query($conn,$GetOldIdSQL);
 
 while($row = mysqli_fetch_array($Query)){
 
 $DefaultId = $row['id'];
 }
 
 $ImagePath = "htdocs/$DefaultId.png";
 
 $ServerURL = "https://kerron.xyz/htdocs/$ImagePath";
 
 $InsertSQL = "insert into kerronxy_users (image_path,image_name) values ('$ServerURL','$ImageName')";
 
 if(mysqli_query($conn, $InsertSQL)){

 file_put_contents($ImagePath,base64_decode($ImageData));

 echo "Your Image Has Been Uploaded.";
 }
 
 mysqli_close($conn);
 }else{
 echo "Not Uploaded";
 }