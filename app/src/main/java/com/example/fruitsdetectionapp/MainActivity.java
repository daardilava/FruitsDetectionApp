package com.example.fruitsdetectionapp;

import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.Nullable;
import com.example.fruitsdetectionapp.ml.FruitsDetection;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, demoTxt, classified, clickHere;
    ImageView imageView, arrowImage;
    Button picture;

    int imageSize = 224; //default image size

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        demoTxt = findViewById(R.id.demoText);
        clickHere = findViewById(R.id.click_here);
        arrowImage = findViewById(R.id.demoArrow);
        classified = findViewById(R.id.classified);

        demoTxt.setVisibility(View.VISIBLE);
        clickHere.setVisibility(View.GONE);
        arrowImage.setVisibility(View.VISIBLE);
        classified.setVisibility(View.GONE);
        result.setVisibility(View.GONE);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //launch camera if we have permission
                if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //request camera permission if we don't have
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode == 1 && resultCode == RESULT_OK){
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            demoTxt.setVisibility(View.GONE);
            clickHere.setVisibility(View.VISIBLE);
            arrowImage.setVisibility(View.GONE);
            classified.setVisibility(View.VISIBLE);
            result.setVisibility(View.VISIBLE);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);

        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
    try {
        FruitsDetection model = FruitsDetection.newInstance(getApplicationContext());
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        //get ID array of 224 * 224 pixels and extract R,G,B values, add to bytebuffer
        int[] intValue = new int[imageSize * imageSize];
        image.getPixels(intValue, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

        //iterate over pixels and extract R,G,B values, add to bytebuffer
        int pixel = 0;
        for (int i = 0; i < imageSize; i++){
            for (int j = 0; j < imageSize; j++){
                int val = intValue[pixel++]; //RGB
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }

        inputFeature0.loadBuffer(byteBuffer);

        // Runs model inference and gets result.
        FruitsDetection.Outputs outputs = model.process(inputFeature0);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

        float[] confidence = outputFeature0.getFloatArray();

        //find the index of the class with the biggest confidence
        int maxPos =  0;
        float maxConfidence = 0;
        for (int i = 0; i < confidence.length; i++){
            if (confidence[i] > maxConfidence){
                maxConfidence = confidence[i];
                maxPos = i;
            }
        }
        String[] classes = {"apple","apricot","avocado","banana","cherimoya","coconut","dewberry","dragonfruit","feijoa","grape","grenadilla","guava","mandarine","mango","mangosteen",
                "morinda","passionfruit","pineapple","pomegranate","orange","orange","strawberry","watermelon","lemon"};
        result.setText(classes[maxPos]);
        result.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // to search the fruit on internet
                startActivity(new Intent(Intent.ACTION_VIEW,
                        Uri.parse("https://www.google.com/search?q="+result.getText())));
            }
        });

        model.close();

    } catch (IOException e){
        // TODO Handle the exception
    }

    }
}












