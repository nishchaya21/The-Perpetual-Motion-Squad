import React from 'react';
import './Banner.scss';
import {images} from '../../assets/images';

// import imageUpload from '../../assets/upload/imageUpload';
// import { BrowserRouter as Router,Route,Routes } from 'react-router-dom';

// import { useNavigate } from 'react-router-dom';

const Banner = () => { 

    // const navigate = useNavigate();

    return(
        <div>
            <div className="banner container">
                <picture>
                    <source media="(max-width:767px)" srcset={images.works_mobile} />
                    <img src={images.works_desktop} alt="" />
                </picture>

                <div className="banner__wrapper">
                    <div className="title">
                        <h2 className="title1">Know about the <br /> damage status of your car!</h2>
                    </div>
                    {/* <div className="button"> */}
                        {/* <button onClick={()=>navigate("/predict")} className='btn'>How do we Work? Just upload image!</button> */}
                        {/* <button onClick={()=>navigate("/predict")} className="btn">Work</button> */}
                    {/* </div> */}
                    <button className='button'>
                        <h3>
                            <a href="http:localhost:8501" className='link'>How do we work?<br></br>Just upload Image!</a>
                        </h3>
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Banner;