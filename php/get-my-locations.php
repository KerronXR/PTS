<?php


$conn = mysqli_connect("78.140.191.36","kerronxy_yacov", "]k2oIl?WBRPh", "kerronxy_users");
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "SELECT lat, lon, image_path FROM `images` ";
$result = $conn->query($sql);
$data = array();
if ($result->num_rows > 0) {

    while($row = $result->fetch_assoc()) {
        array_push($data,$row);
    }
} else {
    echo "0";
}

echo json_encode($data);
$conn->close();
?>

?>