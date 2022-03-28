import React, { useState, useRef } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageURL, setImageURL] = useState(null); // for displaying the image
  const [loaded, setLoaded] = useState(false);

  const fileInputRef = useRef(null);
  const imgRef = useRef(null);

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
    setImageURL(URL.createObjectURL(event.target.files[0]));
    setLoaded(!loaded);
  };

  const predictImage = () => {
    const formData = new FormData();

    formData.append("file", selectedFile);
    formData.append("download_image", true);

    const config = {
      headers: {
        "content-type": "multipart/form-data",
      },
    };

    axios
      .post("http://localhost:8080/detect", formData, config)
      .then((response) => {
        console.log(response.data[0]);
        console.log(response.data[0].length);
        imgRef.current.src = `data:image/jpeg;base64,${response.data[1].image_base64}`;
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const triggerUpload = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="App">
      <h1 className="header">Yolo V5 - Object Detection</h1>
      <div className="inputHolder">
        <input
          type="file"
          accept="image/*"
          capture="camera"
          className="uploadInput"
          onChange={handleFileSelect}
          ref={fileInputRef}
        />
        <button
          className={`${loaded ? "hidden" : "uploadImage"}`}
          onClick={triggerUpload}
        >
          Upload Image
        </button>
        {selectedFile && (
          <button className="buttonz" onClick={predictImage}>
            Identify Image
          </button>
        )}
      </div>

      <div className="mainWrapper">
        <div className="mainContent">
          <div className="imageHolder">
            {imageURL && (
              <img
                src={imageURL}
                alt="Upload Preview"
                crossOrigin="anonymous"
                ref={imgRef}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
