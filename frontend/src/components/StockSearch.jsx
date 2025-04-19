import React, { useState } from "react";
import Select from "react-select";

const StockSearch = ({setSelectedStocks}) => {
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState([]);

  const fetchStocks = async (inputValue) => {
    if (!inputValue) return;

    try {
      const response = await fetch(`http://localhost:${import.meta.env.VITE_PORT}/search?q=${inputValue}`);
      const data = await response.json();

      if (data.results && data.results.quotes) {
        setOptions(
          data.results.quotes.map((stock) => ({
            label: `${stock.shortname} (${stock.symbol})`,
            value: stock.symbol,
          }))
        );
      } else {
        setOptions([]);
      }
    } catch (error) {
      console.error("Error fetching stocks:", error);
    }
  };

  return (
    <div style={{ width: "400px", margin: "50px auto", color: "black" }}>
      <h2 className="text-white">Search for Stocks</h2>
      <Select
        options={options}
        onInputChange={(value) => {
          setQuery(value);
          fetchStocks(value);
        }}
        onChange={(selectedOption) => {
          // Update the selectedStocks state by adding the new selection(s)
          setSelectedStocks((prevState) => {
            // Combine previous state with the new selected options, then filter out duplicates based on the value (stock symbol)
            const updatedStocks = [...prevState, ...selectedOption];
          
            // Remove duplicates by ensuring each stock symbol is unique
            return updatedStocks.filter(
              (stock, index, self) =>
                index === self.findIndex((t) => t.value === stock.value)
            );
          });
          
        }}
        placeholder="Search for a stock..."
        isClearable
        isSearchable
        isMulti
      />
      
    </div>
  );
};

export default StockSearch;
