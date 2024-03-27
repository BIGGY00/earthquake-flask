from flask import Flask, request, send_file, jsonify, make_response
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import os
import numpy as np
import math
from nfft import nfft
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

file_names = [
    "KHLONG_MARUI",
    "MAE_CHAN",
    "MAE_HONG_SON",
    "MAE_ING",
    "MOEI",
    "MAE_THA",
    "THOEN",
    "PHETCHABUN",
    "PUA",
    "PHA_YAO",
    "RANONG",
    "SI_SAWAT",
    "THREE_PAGODA",
    "UTTARADIT",
    "WIANG_HAENG",
    "MAE_LAO"
]

@app.route('/generate_plot_time', methods=['GET'])
def generate_plot_time():
    try:
        # Get the ID from the request arguments
        id = request.args.get('id')

        print("ID received:", id)

        if id is None:
            print("Error: ID is missing in the request.")
            return jsonify({'error': 'ID is missing in the request.'}), 400

        id = int(id)  # Convert id to int

        if not 1 <= id <= len(file_names):
            print("Error: Invalid ID provided.")
            return jsonify({'error': 'Invalid ID provided.'}), 400

        filename = f'data_{id}_{file_names[id-1]}.csv'
        data_path = os.path.join(os.path.dirname(__file__), 'dataV2_13-March-2024', filename)

        print("Data path:", data_path)

        if not os.path.exists(data_path):
            print("Error: Data file not found.")
            return jsonify({'error': 'Data file not found.'}), 404
        
        # Read the data file
        data = pd.read_csv(data_path)

        x = data['RangeDay'].values - data['RangeDay'].values[0] #always start at x=0

        N = len(x)
        if (N%2 != 0):
            N = N - 1
            x = x[:N]

        xh = data['RangeDay'][N//2:].values - data['RangeDay'][N//2:].values[0] #always start at x=0
        # number of N/2 sample points
        Nh = len(xh)
        if (Nh%2 != 0):
            Nh = Nh -1
            xh = xh[-Nh:]

        x_normalise = [i*0.4999/max(x) for i in x]
        xh_normalise = [i*0.4999/max(xh) for i in xh] #fixed bugs

        y = data['Magnitude'][:N].values
        yh = data['Magnitude'][-Nh:].values # using N half-samples 

        yf = np.abs(nfft(x_normalise[:N], y, sigma=5)) #default sigma=5
        amp = 1.0/N * yf 
        yfh = np.abs(nfft(xh_normalise[-Nh:], yh, sigma=5)) #default sigma=5
        amph = 1.0/Nh * yfh

        fig, axs = plt.subplots(1)
        fig_f, axs_f = plt.subplots(1)

        axs.plot(x, y, '.', color='blue')

        axs.set_xlabel('Days')
        axs.set_ylabel('Magnitude')

        xf = np.fft.fftfreq(N,1./N)

        YearFirst = pd.to_datetime(data['Date'][0]).year

        YearEnd = pd.to_datetime(data['Date'].iloc[-1]).year
    
        YearBlue = YearFirst + ((max(x) - max(xh)) / 365)
        if YearBlue % 1 >= 0.5:
            YearBlue = math.ceil(YearBlue)
        else:
            YearBlue = math.floor(YearBlue)

        red_label = f"Year {YearFirst} - {YearEnd}"
        blue_label = f"Year {int(YearBlue)} - {YearEnd}"

        axs_f.plot(xf[:int(N // 2)], amp[:int(N // 2)], color='red', linewidth=1, label=red_label)
        axs_f.plot(xf[:int(Nh // 2)], amph[:int(Nh // 2)], color='blue', linewidth=1, label=blue_label)
        axs_f.set_xlabel('Frequency')
        axs_f.set_ylabel('Amplitude')
        axs_f.legend()
        plt.grid()
        plt.xlim(-1, 58)
        
        # Save the plots as images using PIL
        img_path1 = os.path.join('static', 'images', 'figure1.png')
        img_path2 = os.path.join('static', 'images', 'figure2.png')

        fig.savefig(img_path1, format='png')
        fig_f.savefig(img_path2, format='png')

        # Close the plots to prevent memory leaks
        plt.close(fig)
        plt.close(fig_f)

        # Open images with PIL and send them as responses
        with open(img_path1, 'rb') as img_file1, open(img_path2, 'rb') as img_file2:
            image1 = Image.open(img_file1)
            image2 = Image.open(img_file2)

            # Convert images to bytes
            img_bytes1 = BytesIO()
            image1.save(img_bytes1, format='PNG')
            img_bytes1.seek(0)

            img_bytes2 = BytesIO()
            image2.save(img_bytes2, format='PNG')
            img_bytes2.seek(0)

        response = make_response(img_bytes1)
        response.headers['Content-Type'] = 'image/png'
        
        # Send the first image as response
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_plot_nfft', methods=['GET'])
def generate_plot_nfft():
    try:
        # Get the ID from the request arguments
        id = request.args.get('id')

        print("ID received:", id)

        if id is None:
            print("Error: ID is missing in the request.")
            return jsonify({'error': 'ID is missing in the request.'}), 400

        id = int(id)  # Convert id to int

        if not 1 <= id <= len(file_names):
            print("Error: Invalid ID provided.")
            return jsonify({'error': 'Invalid ID provided.'}), 400

        filename = f'data_{id}_{file_names[id-1]}.csv'
        data_path = os.path.join(os.path.dirname(__file__), 'dataV2_13-March-2024', filename)

        print("Data path:", data_path)

        if not os.path.exists(data_path):
            print("Error: Data file not found.")
            return jsonify({'error': 'Data file not found.'}), 404
        
        # Read the data file
        data = pd.read_csv(data_path)

        x = data['RangeDay'].values - data['RangeDay'].values[0] #always start at x=0

        N = len(x)
        if (N%2 != 0):
            N = N - 1
            x = x[:N]

        xh = data['RangeDay'][N//2:].values - data['RangeDay'][N//2:].values[0] #always start at x=0
        # number of N/2 sample points
        Nh = len(xh)
        if (Nh%2 != 0):
            Nh = Nh -1
            xh = xh[-Nh:]

        x_normalise = [i*0.4999/max(x) for i in x]
        xh_normalise = [i*0.4999/max(xh) for i in xh] #fixed bugs

        y = data['Magnitude'][:N].values
        yh = data['Magnitude'][-Nh:].values # using N half-samples 

        yf = np.abs(nfft(x_normalise[:N], y, sigma=5)) #default sigma=5
        amp = 1.0/N * yf 
        yfh = np.abs(nfft(xh_normalise[-Nh:], yh, sigma=5)) #default sigma=5
        amph = 1.0/Nh * yfh

        fig, axs = plt.subplots(1)
        fig_f, axs_f = plt.subplots(1)

        axs.plot(x, y, '.', color='blue')

        axs.set_xlabel('Days')
        axs.set_ylabel('Magnitude')

        xf = np.fft.fftfreq(N,1./N)

        YearFirst = pd.to_datetime(data['Date'][0]).year

        YearEnd = pd.to_datetime(data['Date'].iloc[-1]).year
    
        YearBlue = YearFirst + ((max(x) - max(xh)) / 365)
        if YearBlue % 1 >= 0.5:
            YearBlue = math.ceil(YearBlue)
        else:
            YearBlue = math.floor(YearBlue)

        red_label = f"Year {YearFirst} - {YearEnd}"
        blue_label = f"Year {int(YearBlue)} - {YearEnd}"

        axs_f.plot(xf[:int(N // 2)], amp[:int(N // 2)], color='red', linewidth=1, label=red_label)
        axs_f.plot(xf[:int(Nh // 2)], amph[:int(Nh // 2)], color='blue', linewidth=1, label=blue_label)
        axs_f.set_xlabel('Frequency')
        axs_f.set_ylabel('Amplitude')
        axs_f.legend()
        plt.grid()
        plt.xlim(-1, 58)
        
        # Save the plots as images using PIL
        img_path1 = os.path.join('static', 'images', 'figure1.png')
        img_path2 = os.path.join('static', 'images', 'figure2.png')

        fig.savefig(img_path1, format='png')
        fig_f.savefig(img_path2, format='png')

        # Close the plots to prevent memory leaks
        plt.close(fig)
        plt.close(fig_f)

        # Open images with PIL and send them as responses
        with open(img_path1, 'rb') as img_file1, open(img_path2, 'rb') as img_file2:
            image1 = Image.open(img_file1)
            image2 = Image.open(img_file2)

            # Convert images to bytes
            img_bytes1 = BytesIO()
            image1.save(img_bytes1, format='PNG')
            img_bytes1.seek(0)

            img_bytes2 = BytesIO()
            image2.save(img_bytes2, format='PNG')
            img_bytes2.seek(0)

        response = make_response(img_bytes2)
        response.headers['Content-Type'] = 'image/png'
        
        # Send the first image as response
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Set threaded=False to run Flask in single-threaded mode
