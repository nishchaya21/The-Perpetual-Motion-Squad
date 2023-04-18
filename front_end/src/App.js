import  React from "react";
import "./scss/main.scss";
import Header from './components/header/Header';
import Footer from './components/footer/Footer';
import Home from "./assets/home/Home";
import Predict from '../src/assets/upload/Predict';

import {BrowserRouter as Router,Route,Routes} from 'react-router-dom';

const App = ()=> {
  return (
    <div>
      <Router>
        <Header />
        <Routes>
          <Route path='/' element={<Home />} /> 
          <Route path='/predict' element={<Predict />} />
        </Routes>
        <Footer />
      </Router>
    </div>
  );
}

export default App;
