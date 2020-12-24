import React, { useEffect, useState } from 'react';
import Canvas from './Canvas';
import styled from 'styled-components';

const Wrapper = styled.div`
  display: grid;
  width: 70%;
  margin: 0 auto;
  align-self: center;
  grid-auto-flow: row;
`;

const Title = styled.h1`
  text-align: center;
`;

const YouWroteText = styled.div`
  text-align: center;
  font-size: 60px;
`;

const ResultText = styled.div`
  text-align: center;
  font-size: 60px;
`;

const App = () => {
  const [showResult, setShowResult] = useState(false);
  const [result, setResult] = useState('');
  const [placeholder, setPlaceholder] = useState('');
  const [shuffling, setShuffling] = useState(false);
  let interval = null;

  useEffect(() => {
    if (shuffling) {
      interval = setInterval(() => {
        setPlaceholder(
          String(Number(placeholder) + Number.parseInt(Math.random() * 9)) % 10
        );
      }, 100);
    } else {
      clearInterval(interval);
    }
  }, [shuffling]);
  return (
    <Wrapper>
      <Title>Digit Classifier</Title>
      <Canvas
        setResult={setResult}
        setShowResult={setShowResult}
        setShuffling={setShuffling}
      />
      {showResult && (
        <ResultText>
          <YouWroteText>You wrote: </YouWroteText>
          <ResultText>{shuffling ? placeholder : result}</ResultText>
        </ResultText>
      )}
    </Wrapper>
  );
};

export default App;
