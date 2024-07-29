package com.software.myapplicationia;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.software.myapplicationia.ml.ModeloFinal;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private ImageView imgFotografia; // Muestra la foto seleccionada o tomada
    private Button btnCamara, btnGaleria; // Botones
    private TextView resultado; // Muestra el resultado de la clasificación
    private static final int CAMARA_REQUEST_CODE = 100; // Código de solicitud para la cámara
    private static final int GALERIA_REQUEST_CODE = 1000; // Código de solicitud para la galería
    private int imageSize = 200; // Tamaño de entrada del modelo
    private float confidenceThreshold = 0.6f; // Umbral de confianza para la clasificación

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgFotografia = findViewById(R.id.imageView); // Vincula el ImageView para mostrar la foto.
        btnCamara = findViewById(R.id.btnTomarFoto); // Botón para activar la cámara.
        btnGaleria = findViewById(R.id.btnGaleria); // Botón para seleccionar una foto de la galería.
        resultado = findViewById(R.id.result); // TextView para mostrar el resultado de la clasificación.

        // Solicitar permisos si no están concedidos
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
                checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
        }

        // Acción para el botón de la cámara
        btnCamara.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Inicia la cámara para capturar una foto
                Intent camara = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(camara, CAMARA_REQUEST_CODE);
            }
        });

        // Acción para el botón de la galería
        btnGaleria.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Inicia la galería para seleccionar una foto
                Intent galeria = new Intent(Intent.ACTION_PICK);
                galeria.setData(MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galeria, GALERIA_REQUEST_CODE);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Bitmap imageBitmap = null;

            try {
                // Verifica si la imagen proviene de la cámara o la galería
                if (requestCode == CAMARA_REQUEST_CODE) {
                    // Obtiene la imagen de la cámara
                    Bundle extras = data.getExtras();
                    imageBitmap = (Bitmap) extras.get("data");
                } else if (requestCode == GALERIA_REQUEST_CODE) {
                    // Obtiene la imagen de la galería
                    Uri selectedImageUri = data.getData();
                    imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
                }

                if (imageBitmap != null) {
                    // Redimensionar la imagen para que coincida con el tamaño de entrada del modelo
                    imageBitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false);
                    imgFotografia.setImageBitmap(imageBitmap); // Mostrar la imagen en el ImageView
                    clasificarImagen(imageBitmap); // Clasificar la imagen usando el modelo
                } else {
                    Toast.makeText(this, "Error al cargar la imagen", Toast.LENGTH_SHORT).show();
                }
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Error al procesar la imagen", Toast.LENGTH_SHORT).show();
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Ocurrió un error inesperado", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // Método para clasificar la imagen usando el modelo de TensorFlow Lite
    public void clasificarImagen(Bitmap imagen) {
        try {
            // Cargar el modelo de TensorFlow Lite
            ModeloFinal modelo = ModeloFinal.newInstance(getApplicationContext());

            // Crear TensorBuffer para entrada de datos con el tamaño especificado
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Convertir los píxeles de la imagen a float y normalizar de 0-255 a 0-1
            int[] valoresInt = new int[imageSize * imageSize];
            imagen.getPixels(valoresInt, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());
            int pixel = 0;

            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = valoresInt[pixel++];
                    // Extrae y normaliza los valores de los canales de color
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer); // Carga los datos en el TensorBuffer

            // Ejecutar la inferencia con el modelo
            ModeloFinal.Outputs resultados = modelo.process(inputFeature0);
            TensorBuffer outputFeature0 = resultados.getOutputFeature0AsTensorBuffer();

            float[] confianzas = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfianza = 0;

            // Encuentra la clase con la mayor confianza
            for (int i = 0; i < confianzas.length; i++) {
                if (confianzas[i] > maxConfianza) {
                    maxConfianza = confianzas[i];
                    maxPos = i;
                }
            }

            // Definir las clases según el modelo entrenado
            String[] clases = {
                    "Maíz - Cercospora",
                    "Maíz - Óxido común",
                    "Maíz - Saludable",
                    "Maíz - Tizón de la hoja del norte",
                    "Manzana - Costra de Manzana",
                    "Manzana - Óxido de cedro",
                    "Manzana - Pedumbre Negra",
                    "Manzana - Saludable",
                    "Uva - Mancha foliar de Isariopsis",
                    "Uva - Podredumbre negra",
                    "Uva - Saludable",
                    "Uva - Sarampión negro"
            };

            // Verificar si la confianza es suficiente
            if (maxConfianza > confidenceThreshold) {
                resultado.setText(clases[maxPos]); // Mostrar la clase predicha
            } else {
                resultado.setText("No se reconoce la clase con suficiente confianza");
            }

            modelo.close(); // Cerrar el modelo para liberar recursos
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error al cargar el modelo", Toast.LENGTH_SHORT).show();
        }
    }
}
