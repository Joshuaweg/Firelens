# Firelens
This is a PyTorch Machine Learning Project where satellite images are analyzed to determine the presence of smoke and wildfires

when you have downloaded the repository I recommend:

    create a virtual environment
    run the virtual environment
    run the 'command pip install -r requirements.txt' to install dependencies

When that is all done, you can run the project with 'python app.py'

Note is you get the following error:

![image](https://github.com/Joshuaweg/Firelens/assets/21377489/3bd044cc-b5a4-4abb-ba9f-6ac295903440)

It is because the Captum library has a bug in its code that may need to be updated manually. please follow the instructions from the Captum repository posted here to correct this bug:

![image](https://github.com/Joshuaweg/Firelens/assets/21377489/1603e0ca-18d2-45e7-aad8-81a1a7488304)

expected output when app.py is working:

![image](https://github.com/Joshuaweg/Firelens/assets/21377489/fb091e52-8b6e-492c-b35d-5948f1cb9720)


Activation Map using integrated Gradients:
![image](https://github.com/Joshuaweg/Firelens/assets/21377489/4f5af438-a25d-4050-90fb-db121816f125)
