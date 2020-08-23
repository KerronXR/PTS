<?php

require_once 'connect.php';

/*
 test if success log
 $email = "demoaccount@gmail.com";
$password = "123456";
 */
    $email = $_POST['email'];
    $password = $_POST['password'];



    $isVaildEmail = filter_var($email, FILTER_VALIDATE_EMAIL);
    if($conn){
        if($isVaildEmail === false){
            echo "This Email is not valid";
        }else{
            $sqlCheckEmail = "SELECT * FROM  users_table WHERE email = '$email'";
            $emailQuery = mysqli_query($conn, $sqlCheckEmail);


            if (mysqli_num_rows($emailQuery) >0){
                $sqlLogin = "SELECT * FROM  users_table WHERE email = '$email' AND  password = '$password'";
                $loginQuery = mysqli_query($conn, $sqlLogin);

                if(mysqli_num_rows($loginQuery) >0){
                    echo "Login success";
                }else{
                    echo "Wrong password";
                }
            }else{
                echo "This Email is not registered";
            }
        }

    }else{
        echo  "Connection Error";
    }







