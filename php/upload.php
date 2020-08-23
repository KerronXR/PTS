<?php

$conn = mysqli_connect("78.140.191.36","kerronxy_yacov", "]k2oIl?WBRPh", "kerronxy_users");


$target_dir = "uploads/";
$target_file = $target_dir . basename($_FILES["image"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image
if(isset($_POST["submit"])) {
    $check = getimagesize($_FILES["image"]["tmp_name"]);
    if($check !== false) {
        echo "File is an image - " . $check["mime"] . ".";
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}
// Check if file already exists
if (file_exists($target_file)) {
    echo "Sorry, file already exists.";
    $uploadOk = 0;
}

// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    echo "Sorry, your file was not uploaded.";
// if everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["image"]["tmp_name"], $target_file)) {
        $sql = 'INSERT INTO `images`( `image_name`, `image_path`, `user_email`, `lat`, `lon`) VALUES 
        ("lat'.$_REQUEST['lat'].'lon'.$_REQUEST['lon'].'image'.basename($_FILES["image"]["name"]).'", "https://kerron.xyz/htdocs/uploads/'.basename($_FILES["image"]["name"]).'","'.$_REQUEST['email'].'", "'.$_REQUEST['lat'].'",
        "'.$_REQUEST['lon'].'")';
        if ($conn->query($sql) === TRUE) {
            echo "1";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    } else {
        echo "Sorry, there was an error uploading your file.";
    }
}
?>