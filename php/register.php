<?php


require "connect.php";

$fullname = $_POST['fullname'];
$username = $_POST['username'];
$email = $_POST['email'];
$password = $_POST['password'];
$gender = $_POST['gender'];


$isVaildEmail = filter_var($email, FILTER_VALIDATE_EMAIL);

if($conn){
    if(strlen($password) > 40 ||strlen($password) <6){
        echo "Password must be less than 40 and more than 6 characters";
    } elseif ($isVaildEmail === false){
        echo "The email is not valid";
    }else{
        $sqlCheckUsername = "SELECT * FROM  users_table WHERE username LIKE '$username'";
        $usernameQuery = mysqli_query($conn, $sqlCheckUsername);

        $sqlCheckEmail = "SELECT * FROM  users_table WHERE email LIKE '$email'";
        $emailQuery = mysqli_query($conn, $sqlCheckEmail);

        if(mysqli_num_rows($usernameQuery) >0){
            echo "User name is already in use, try another one";
        }elseif (mysqli_num_rows($emailQuery) >0){
            echo "This Email already registered, Try another Email";
        }
        else{
            $sql_register = "INSERT INTO users_table(fullname, username, email, password, gender) VALUES ('$fullname', '$username' , '$email', '$password', '$gender')";

            if(mysqli_query($conn, $sql_register)){
                echo "Successfully Registered";
            }else{
                echo "Failed to Register";
            }
        }
    }

}


