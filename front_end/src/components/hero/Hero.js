import React from 'react';
import './Hero.scss';
import {images} from '../../assets/images';

const Hero = () => {
    return(
        <div>
            <div className='hero'>
                <div className='right-top-image'>
                    <picture>
                        <source media="(max-width:767px)" srcSet={images.intro_right_mobile}/>
                        <img src={images.intro_right} alt="" />
                    </picture>
                </div>
                <div className='hero__wrapper container'>
                    <div className='hero__content'>
                        <h1 className='title1'>
                        We have good Brakes. <br /> Do you have good insurance?
                        </h1>
                        <p className='hero__text'>
                            Ensuring your future dreams. 
                            Get your car insurance coverage easier and faster. 
                            We blend our expertise and technology to help you 
                            find the plan that is right and the best for you and your family.
                        </p>
                        <button className='btn'>VIEW PLANS</button>
                    </div>
                    <div className="hero__image">
                        <picture>
                            <source media="(max-width:767px)" srcSet={images.intro_mobile} />
                            <img src={images.intro_desktop} alt="" />
                        </picture>
                    </div>
                </div>
                <div className='left-bottom-image'>
                    <picture>
                        <source media="(max-width:767px)" srcSet={images.intro_left_mobile} />
                        <img src={images.intro_left} alt="" />
                    </picture>
                </div>
            </div>
        </div>
    )
}

export default Hero;