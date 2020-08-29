package com.example.potholetrackingsystem;

import android.Manifest;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.icu.text.DateFormat;
import android.location.Criteria;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.NetworkResponse;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.Volley;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;

import java.io.ByteArrayOutputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class UserUploadImage extends AppCompatActivity implements LocationListener {

//    LocationFinder locationFinder;
    Button CaptureImageFromCamera, UploadImageToServer;
    ImageView ImageViewHolder;
    public static final int RequestPermissionCode = 1;
    String ImageUploadPathOnServer = "https://kerron.xyz/htdocs/upload.php";
    Bitmap selected_bitmap;
    ProgressDialog dialog;
    RequestQueue requestQueue;
    LocationManager manager;
    boolean found = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_uploadimage);
//        locationFinder = new LocationFinder(this);
        CaptureImageFromCamera = findViewById(R.id.btn_select_image);
        ImageViewHolder = findViewById(R.id.imageView);
        UploadImageToServer = findViewById(R.id.btn_upload_image);

        CaptureImageFromCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (ActivityCompat.checkSelfPermission(UserUploadImage.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent intent_ = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(intent_, 7);
                } else {
                    ActivityCompat.requestPermissions(UserUploadImage.this, new String[]{Manifest.permission.CAMERA}, 101);
                }

            }
        });

        UploadImageToServer.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                LocationMethod();
            }
        });
    }

    // Start activity for result method to Set captured image on image view after click.
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 7 && resultCode == RESULT_OK) {
            selected_bitmap = (Bitmap) data.getExtras().get("data");
            ((ImageView) findViewById(R.id.imageView)).setImageBitmap(selected_bitmap);
        }

    }

    @Override
    public void onRequestPermissionsResult(int RC, String per[], int[] PResult) {

        if (RC == RequestPermissionCode) {
            if (PResult.length > 0 && PResult[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(UserUploadImage.this, "Permission Granted, Now your application can access CAMERA.", Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(UserUploadImage.this, "Permission Canceled, Now your application cannot access CAMERA.", Toast.LENGTH_LONG).show();
            }
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    void LocationMethod() {

        LocationManager lm = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
        boolean gps_enabled = false;
        boolean network_enabled = false;

        try {
            gps_enabled = lm.isProviderEnabled(LocationManager.GPS_PROVIDER);
        } catch (Exception ex) {
        }

        try {
            network_enabled = lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER);
        } catch (Exception ex) {
        }

        if (!gps_enabled && !network_enabled) {
            // notify user
            new AlertDialog.Builder(UserUploadImage.this)
                    .setMessage("Location is not enabled. Please enable location to continue.")
                    .setPositiveButton("Open Setting", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface paramDialogInterface, int paramInt) {
                            startActivity(new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS));

                        }
                    }).setNegativeButton("Cancel", null)
                    .show();
        } else {

            dialog = new ProgressDialog(UserUploadImage.this);
            dialog.setTitle("Location");
            dialog.setMessage("Please wait while getting current location...");
            dialog.show();
            get_current_location();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    void get_current_location() {
        manager = (LocationManager) getSystemService(LOCATION_SERVICE);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(UserUploadImage.this, "Please allow access to you location", Toast.LENGTH_SHORT).show();
            dialog.dismiss();
            requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION}, 4);
        } else {

            FusedLocationProviderClient fusedLocationClient;
            fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
            fusedLocationClient.getLastLocation()
                    .addOnSuccessListener(this, new OnSuccessListener<Location>() {
                        @Override
                        public void onSuccess(Location location) {
                            if (location != null) {

                                found = true;
                                double latitude = location.getLatitude();
                                double longitude = location.getLongitude();

                                save_location(latitude, longitude);

                                dialog.dismiss();
                                //Toast.makeText(StartTask.this, location.getLatitude() + "", Toast.LENGTH_SHORT).show();
                            } else {
                                Criteria crit = new Criteria();
                                crit.setAccuracy(Criteria.ACCURACY_FINE);
                                if (ActivityCompat.checkSelfPermission(UserUploadImage.this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                                    return;
                                }
                                manager.requestLocationUpdates(manager.getBestProvider(crit, false), 1000, 1, UserUploadImage.this);
                            }
                        }
                    }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    dialog.dismiss();
                    Toast.makeText(UserUploadImage.this, "Error Occur while getting current location", Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    @Override
    public void onLocationChanged(Location location) {
        dialog.dismiss();
        if (!found) {
            found = true;
            double latitude = location.getLatitude();
            double longitude = location.getLongitude();
            dialog.dismiss();
            save_location(latitude, longitude);
        }
    }

    void save_location(double lat, double lon) {
        if (selected_bitmap != null)
            uploadImage(lat, lon);
        else Toast.makeText(UserUploadImage.this, "Please select Image to continue", Toast.LENGTH_SHORT).show();
    }

    private void uploadImage(final double lat, final double lon) {

        if (lat == 0.0 || lon == 0.0) {
            Toast.makeText(UserUploadImage.this, "Please enable location to upload image", Toast.LENGTH_SHORT).show();
            return;
        }

        final ProgressDialog dialog = new ProgressDialog(UserUploadImage.this);
        dialog.setMessage("Uploading Image...");
        dialog.setCancelable(false);
        dialog.show();
        VolleyMultipartRequest volleyMultipartRequest = new VolleyMultipartRequest(Request.Method.POST, ImageUploadPathOnServer,
                new Response.Listener<NetworkResponse>() {
                    @Override
                    public void onResponse(NetworkResponse response) {

                        if (new String(response.data).equals("1")) {
                            Toast.makeText(UserUploadImage.this, "Image has been uploaded", Toast.LENGTH_SHORT).show();
                        }
                        dialog.dismiss();
                        Toast.makeText(UserUploadImage.this, "" + new String(response.data), Toast.LENGTH_SHORT).show();
                        requestQueue.getCache().clear();


                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        dialog.dismiss();
                        Toast.makeText(UserUploadImage.this, error.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                }) {

            /*
             * If you want to add more parameters with the image
             * you can do it here
             * here we have only one parameter with the image
             * which is tags
             * */
            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                Map<String, String> params = new HashMap<>();

                params.put("email", getSharedPreferences("user", MODE_PRIVATE).getString("id", ""));
                params.put("lat", lat + "");
                params.put("lon", lon + "");
                return params;
            }

            /*
             *pass files using below method
             * */
            @Override
            protected Map<String, DataPart> getByteData() {
                Map<String, DataPart> params = new HashMap<>();
                long current_time_in_millis = System.currentTimeMillis();
                String current_time = String.valueOf(current_time_in_millis);
                params.put("image", new VolleyMultipartRequest.DataPart(current_time + "." + (new Random().nextInt(1000000))
                        + ".png", getFileDataFromDrawable(selected_bitmap)));
                return params;
            }
        };


        volleyMultipartRequest.setRetryPolicy(new DefaultRetryPolicy(
                0,
                DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        requestQueue = Volley.newRequestQueue(UserUploadImage.this);
        requestQueue.add(volleyMultipartRequest);
    }

    public byte[] getFileDataFromDrawable(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
        return byteArrayOutputStream.toByteArray();
    }

}
