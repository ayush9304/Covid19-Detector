
CoviScan
<img src="static/full_brand_logo.png" alt="coviscan" border="0">

Coviscan is an automated imaging tool which process chest x-ray images and predicts whether the person has Covid-19 disease or Pneumonia or Normal.

It uses deep learning algorithm Convolution Neural Networks (CNNs) also known as ConvNets to process and extract features from X-Ray images. The whole pipeline involves several steps like preprocessing, semantic segmentation, classification etc.

## How to use

- **Install Python**
  - Download and install python 3.9 from [python.org](https://www.python.org)


- **Get the code**
  ```
  git clone https://github.com/ayush9304/covid19-detector-api
  ```
 
- **Install Python dependencies**
    ```
    pip3 install -r requirements.txt
    ```
 
- **Run**
    ```
    py manage.py runserver
    ```
      
- **Explore**
  - Goto ```http://127.0.0.1:8000/``` url on any web browser

## License

Licensed under the MIT License.
