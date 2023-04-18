import React from 'react';
import './Info.scss';
import {images} from '../../assets/images';

const Info = () => {
    return(
        <div>
            <div className="we-are-different">
                <div className="container">
                    <h2 className="title1 title1--mod">We are different</h2>
                    <div className="we-are-different__content">
                        <div className="snappy">
                            <img src={images.snappy_process} alt="" />
                            <h3>Snappy Process</h3>
                            <p>
                                Our application process can be completed in minutes, not hours.
                                Don't get stuck filling in tedious forms.
                            </p>
                        </div>
                        <div className="affordable">
                            <img src={images.affordable_prices} alt="" />
                            <h3>Affordable Prices</h3>
                            <p>
                                We don't want you worrying about high monthly costs. 
                                Our prices may be low, but we still offer the best coverage possible.
                            </p>
                        </div>
                        <div className="people">
                            <img src={images.people_first} alt="" />
                            <h3>People First</h3>
                            <p>
                                Our plans aren't full of conditions and clauses to prevent payouts. 
                                We make sure you're covered when you need it.
                            </p>
                        </div>
                    </div>
                </div> 
            </div>
        </div>
    )
}

export default Info;