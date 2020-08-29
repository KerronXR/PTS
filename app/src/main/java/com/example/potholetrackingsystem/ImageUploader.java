package com.example.potholetrackingsystem;

import android.content.Context;
import android.graphics.Bitmap;
import android.icu.text.DateFormat;
import android.os.Build;
import android.widget.Toast;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import com.android.volley.*;
import com.android.volley.toolbox.Volley;

import java.io.ByteArrayOutputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class ImageUploader extends AppCompatActivity {

    private final Context context;
    private final String ImageUploadPathOnServer = "https://kerron.xyz/htdocs/upload.php";
    private Bitmap selected_bitmap = null;
    RequestQueue requestQueue;

    public ImageUploader(Context context) {
        this.context = context;
    }

    public byte[] getFileDataFromDrawable(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
        return byteArrayOutputStream.toByteArray();
    }

    public void uploadImage(final double[] loc, Bitmap bitmap) {
        final double lat = loc[0];
        final double lon = loc[1];
        setSelected_bitmap(bitmap);
        if (lat == 0.0 || lon == 0.0) {
            runToast("Please enable location to upload image");
            return;
        }

        VolleyMultipartRequest volleyMultipartRequest = new VolleyMultipartRequest(Request.Method.POST, ImageUploadPathOnServer,
                new Response.Listener<NetworkResponse>() {
                    @Override
                    public void onResponse(final NetworkResponse response) {
                        if (new String(response.data).equals("1")) {
                            runToast("Image has been uploaded");
                        }
                        requestQueue.getCache().clear();
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        runToast("Error: " + error.getMessage());
                    }
                }) {

            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                Map<String, String> params = new HashMap<>();
                params.put("email", context.getSharedPreferences("user", Context.MODE_PRIVATE).getString("id", ""));
                params.put("lat", lat + "");
                params.put("lon", lon + "");
                return params;
            }

            @RequiresApi(api = Build.VERSION_CODES.N)
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
                1000,
                DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        requestQueue = Volley.newRequestQueue(context);
        requestQueue.add(volleyMultipartRequest);
    }

    private void setSelected_bitmap(Bitmap selected_bitmap) {
        this.selected_bitmap = selected_bitmap;
    }

    private void runToast(final String message) {
        new Thread() {
            public void run() {
                runOnUiThread(new Runnable() {
                    public void run() {
                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }.start();
    }
}
