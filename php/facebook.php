<?php 
// Create connection
$conn = mysqli_connect("78.140.191.36","kerronxy_yacov", "]k2oIl?WBRPh", "kerronxy_users");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "SELECT * FROM `users_table` WHERE email LIKE '".$_REQUEST['id']."'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    echo $_REQUEST['id'];
} else {
    $sql = "INSERT INTO `users_table`(`fulname`, `username`, `email`, `password`, `gender`) VALUES ('".$_REQUEST['name']."','','".$_REQUEST['id']."','','')";
    
    if ($conn->query($sql) === TRUE) {
        echo $_REQUEST['id'];
    } else {
        echo $conn->error;
    }
}
$conn->close();

?>