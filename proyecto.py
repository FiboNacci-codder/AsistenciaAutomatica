import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import os
from datetime import datetime
import csv
import pandas as pd

class VideoApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Aplicación de Reconocimiento Facial")
        self.window.geometry("1000x500")  # Reducida la altura total
        self.window.resizable(False, False)

        # Diccionario para almacenar la primera detección de cada persona
        self.first_detection = {}

        # Frame izquierdo (más pequeño)
        self.left_frame = tk.Frame(window, width=150, height=500)
        self.left_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.left_frame.pack_propagate(False)

        self.record_button = tk.Button(self.left_frame, text="Grabar", command=self.toggle_record)
        self.record_button.pack(pady=5)

        self.show_button = tk.Button(self.left_frame, text="Mostrar Asistencia", command=self.show_attendance)
        self.show_button.pack(pady=5)

        # Frame central (cámara)
        self.center_frame = tk.Frame(window, width=400, height=500)
        self.center_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.center_frame.pack_propagate(False)

        self.video_label = tk.Label(self.center_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Frame derecho (tabla)
        self.right_frame = tk.Frame(window, width=450, height=500)
        self.right_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.right_frame.pack_propagate(False)

        # Modificar la tabla para incluir la fecha y ajustar el ancho de las columnas
        self.tree = ttk.Treeview(self.right_frame, columns=('Nombre', 'Hora', 'Fecha'), show='headings')
        self.tree.heading('Nombre', text='Nombre')
        self.tree.heading('Hora', text='Hora')
        self.tree.heading('Fecha', text='Fecha')
        self.tree.column('Nombre', width=150)
        self.tree.column('Hora', width=100)
        self.tree.column('Fecha', width=100)
        self.tree.pack(expand=True, fill=tk.BOTH)

        # Agregar una barra de desplazamiento vertical
        scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.is_recording = False
        self.cap = None
        
        # Inicializar FaceNet
        self.facenet = FaceNet()
        
        # Inicializar el detector de rostros de OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Crear base de datos de embeddings
        self.database = self.create_database()

        # Diccionario para almacenar la última hora de detección de cada persona
        self.last_detection = {}

    def generate_attendance_csv(self):
        # Leer el CSV de ejecución
        ejecucion_df = pd.read_csv('C:\\Users\\fibon\\Desktop\\UNSAAC\\IX-Ciclo\\DeepLearning\\proyecto\\ejecucion.csv')
        
        # Leer el CSV de alumnos
        alumnos_df = pd.read_csv('C:\\Users\\fibon\\Desktop\\UNSAAC\\IX-Ciclo\\DeepLearning\\proyecto\\alumnos.csv')
        
        # Crear nuevas columnas 'presente', 'hora' y 'fecha' inicializadas con valores por defecto
        alumnos_df['presente'] = 'No'
        alumnos_df['hora'] = ''
        alumnos_df['fecha'] = ''
        
        # Marcar como 'Si' a los alumnos que están en el CSV de ejecución y añadir hora y fecha
        for _, row in ejecucion_df.iterrows():
            nombre = row['Nombre']
            hora = row['Hora']
            fecha = row['Fecha']
            mask = alumnos_df['apellidos_nombres'].str.contains(nombre, case=False)
            alumnos_df.loc[mask, 'presente'] = 'Si'
            alumnos_df.loc[mask, 'hora'] = hora
            alumnos_df.loc[mask, 'fecha'] = fecha
        
        # Guardar el resultado en un nuevo CSV
        alumnos_df.to_csv('C:\\Users\\fibon\\Desktop\\UNSAAC\\IX-Ciclo\\DeepLearning\\proyecto\\asistencia.csv', index=False)

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def save_to_csv(self):
        with open('C:\\Users\\fibon\\Desktop\\UNSAAC\\IX-Ciclo\\DeepLearning\\proyecto\\ejecucion.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Nombre', 'Hora', 'Fecha'])  # Encabezados
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                writer.writerow(values)

    def toggle_record(self):
        if not self.is_recording:
            self.start_record()
        else:
            self.stop_record()

    def start_record(self):
        self.is_recording = True
        self.record_button.config(text="Parar")
        self.show_button.config(state=tk.DISABLED)
        self.cap = cv2.VideoCapture(0)
        self.clear_table()  # Limpia la tabla al iniciar la grabación
        self.first_detection = {}  # Reinicia el diccionario de primera detección
        self.last_detection = {}  # Reinicia el diccionario de última detección
        self.show_frame()

    def stop_record(self):
        self.is_recording = False
        self.record_button.config(text="Grabar")
        self.show_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
        self.video_label.config(image='')
        self.save_to_csv()  # Guarda los datos en CSV al detener la grabación
        self.generate_attendance_csv()  # Genera el CSV de asistencia

    def preprocess_face(self, face):
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)
        return face

    def create_database(self):
        database = {}
        image_dir = "C:\\Users\\fibon\\Desktop\\UNSAAC\\IX-Ciclo\\DeepLearning\\proyecto\\imagenes"
        for filename in os.listdir(image_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face = image[y:y+h, x:x+w]
                    face_preprocessed = self.preprocess_face(face)
                    embedding = self.facenet.embeddings(face_preprocessed)[0]
                    database[name] = embedding
        return database

    def recognize_face(self, embedding):
        min_dist = 0.25  # Umbral de similitud
        identity = None
        for name, db_embedding in self.database.items():
            dist = cosine(embedding, db_embedding)
            if dist < min_dist:
                min_dist = dist
                identity = name
        return identity

    def update_table(self, name):
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        current_time_str = current_time.strftime("%H:%M:%S")
        
        if name not in self.first_detection:
            # Primera aparición
            self.first_detection[name] = current_time
            self.tree.insert('', 'end', values=(name, current_time_str, current_date))
        elif (current_time - self.last_detection[name]).total_seconds() > 60:
            # Ha pasado más de un minuto desde la última detección
            self.tree.insert('', 'end', values=(name, current_time_str, current_date))

        self.last_detection[name] = current_time

        # Mantener solo las últimas 10 entradas
        if len(self.tree.get_children()) > 10:
            self.tree.delete(self.tree.get_children()[0])

    def show_frame(self):
        if self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Preprocesar y obtener el embedding del rostro
                    face_preprocessed = self.preprocess_face(face_img)
                    embedding = self.facenet.embeddings(face_preprocessed)[0]
                    
                    # Reconocer el rostro
                    identity = self.recognize_face(embedding)
                    
                    if identity:  # Solo procesar si se reconoce la cara
                        # Actualizar la tabla
                        self.update_table(identity)
                        
                        # Dibujar rectángulo y nombre
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, identity, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                # Convertir la imagen para mostrarla en tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = img.resize((380, 380), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            self.window.after(10, self.show_frame)

    def show_attendance(self):
        # Limpiar la tabla del frame derecho
        self.clear_table()
        
        # Leer el CSV de asistencia
        try:
            df = pd.read_csv('C:\\Users\\fibon\\Desktop\\UNSAAC\\IX-Ciclo\\DeepLearning\\proyecto\\asistencia.csv')
            
            # Crear una nueva ventana para mostrar el CSV
            attendance_window = tk.Toplevel(self.window)
            attendance_window.title("Asistencia")
            attendance_window.geometry("800x600")  # Aumentado el tamaño para acomodar más columnas
            
            # Crear un widget Treeview para mostrar el contenido del CSV
            tree = ttk.Treeview(attendance_window, columns=list(df.columns), show='headings')
            
            # Configurar las columnas
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)  # Ajusta este valor según sea necesario
            
            # Insertar los datos
            for _, row in df.iterrows():
                tree.insert('', 'end', values=list(row))
            
            # Añadir barras de desplazamiento
            v_scrollbar = ttk.Scrollbar(attendance_window, orient="vertical", command=tree.yview)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar = ttk.Scrollbar(attendance_window, orient="horizontal", command=tree.xview)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            tree.pack(expand=True, fill=tk.BOTH)
            
        except FileNotFoundError:
            tk.messagebox.showerror("Error", "El archivo asistencia.csv no se encuentra.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()