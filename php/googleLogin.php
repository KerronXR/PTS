<?php 
// Create connection
$conn = mysqli_connect("78.140.191.36","kerronxy_yacov", "]k2oIl?WBRPh", "kerronxy_users");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
	echo "Error";
}

$sql = "SELECT * FROM `users_table` WHERE email LIKE '".$_REQUEST['email']."'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while($row = $result->fetch_assoc()) {
        echo $row["email"] . " logged in";
    }
} else {
    
    $sql = "INSERT INTO `users_table`(`fullname`, `username`, `email`, `password`, `gender`) VALUES ('".$_REQUEST['name']."','','".$_REQUEST['email']."','','')";
    
    if ($conn->query($sql) === TRUE) {
        $last_id = $conn->insert_id;
        echo $_REQUEST['email'];
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}
$conn->close();
?>