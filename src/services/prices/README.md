# janas-prices

Prices and finance for `janas-chat` or any other MCP client (see [the services](../README.md)), for information only: what a currency is worth, a crypto-asset, a share, inflation, a central bank's rates, the wholesale price of electricity, the cheapest fuel near a place. Open data without a key, for as many countries as each kind of data covers; only shares need a free key of yours.

## The tools

- **`prices_currency`** - an amount (`amount`, 1 unless asked) from a currency (`from`, EUR unless asked) into others (`to`: USD,GBP,CHF,JPY unless asked), at the ECB's reference rates of the latest working day or of a day (`date`).
- **`prices_crypto`** - a crypto-asset's price (`coin`: a name or a symbol), in a currency (`currency`, EUR unless asked), its change in 24 hours and its market value.
- **`prices_stock`** - a share's last price (`share`: a symbol, AAPL or TSCO.LON, or a company's name), its change, the day's open, high and low, the previous close; with a key of Alpha Vantage only.
- **`prices_inflation`** - a country's inflation (`country`: an ISO code, EA for the euro area; the user's country unless asked): the last thirteen months for the European Union, Iceland and Norway, the last years for every other country.
- **`prices_rates`** - a central bank's rates (`bank`: `ecb`, `fed`, `boe`, `snb`; the user's country's unless asked): the ECB's deposit facility, main refinancing operations, marginal lending facility and €STR, the Fed's effective federal funds rate and target range, the Bank of England's Bank Rate, the Swiss National Bank's policy rate.
- **`prices_electricity`** - the wholesale price of electricity for the next day, hour by hour, for a place's bidding zone (`place`, where the user is unless asked), today or tomorrow (`day`): the price on the exchange, before taxes and charges.
- **`prices_fuel`** - the cheapest stations within a radius of a place (`radius`, 5 km unless asked, 20 at most) for a fuel (`fuel`: petrol, diesel, lpg, cng), their prices, how far and which way, when the price was told; how many stations there are and the area's average.

## The answers

Every answer is data with a layout (see [the services](../README.md)): `janas-chat` shows the user the whole of it in the user's language - every rate, month, hour or station - and gives the model only a brief, from which it answers in a line or two. Every answer says where its figures come from, their date, and that they are for information only, not advice to buy or sell.

## Where the data comes from

| Data | Source | Countries | Key |
|---|---|---|---|
| currencies | the [European Central Bank](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)'s reference rates, through [Frankfurter](https://frankfurter.dev); the ECB's own file when Frankfurter does not answer | the thirty or so currencies the ECB quotes | none |
| crypto-assets | [CoinGecko](https://www.coingecko.com/en/api)'s keyless API (data provided by CoinGecko); [Coinbase](https://www.coinbase.com)'s spot price when it does not answer | - | none |
| shares | [Alpha Vantage](https://www.alphavantage.co) | the exchanges it covers | a free one of yours |
| inflation | [Eurostat](https://ec.europa.eu/eurostat)'s harmonised index of consumer prices (monthly), the [World Bank](https://data.worldbank.org) (yearly, CC BY 4.0) | Eurostat: the EU, Iceland, Norway, the euro area; the World Bank: every other | none |
| central banks | the [ECB](https://data.ecb.europa.eu), the [New York Fed](https://www.newyorkfed.org/markets/reference-rates/effr), the [Bank of England](https://www.bankofengland.co.uk/boeapps/database/), the [Swiss National Bank](https://data.snb.ch) | euro area, United States, United Kingdom, Switzerland | none |
| electricity | [energy-charts.info](https://www.energy-charts.info) (Fraunhofer ISE) | the bidding zones it gives under CC BY 4.0 | none |
| fuel | Italy: the [Ministry of Enterprises (MIMIT)](https://www.mimit.gov.it/it/open-data/elenco-dataset/carburanti-prezzi-praticati-e-anagrafica-degli-impianti) (IODL 2.0), a day's prices; France: [prix-carburants](https://data.economie.gouv.fr/explore/dataset/prix-des-carburants-en-france-flux-instantane-v2/), live; Spain: the [Ministry for the Ecological Transition](https://energia.serviciosmin.gob.es/ServiciosRestCarburantes/PreciosCarburantes/help); Austria: [E-Control](https://www.e-control.at)'s fuel price calculator | Italy, France, Spain, Austria | none |
| places | [Nominatim](https://nominatim.org) (OpenStreetMap), the built-in tables when it does not answer | - | none |

- **Electricity only where it is open data.** energy-charts gives with each zone's prices their licence: the zones whose prices come from the Bundesnetzagentur's SMARD under CC BY 4.0 are given (on 2 October 2026: Germany-Luxembourg, Austria, Belgium, France, the Netherlands, northern Italy, among those tried); the others' are for private and internal use only, and the answer says they are not open data (Sicily, Spain, among those tried). The zone of an Italian place is its region's; that of a Nordic country's place is guessed from the point.
- **Fuel is as the stations tell it.** Italy's prices are those of the morning file, each with the time the station told it; France's are live, and some are weeks old (their date says so); Spain's are the ministry's of the moment; Austria's are those E-Control gives for a point: some of the stations near it, the cheapest. The stations' addresses are as their operators gave them. No other country publishes every station's prices as open data that `janas-prices` reads: the answer says so.
- **The sources' rules are kept.** A User-Agent naming the program, a pause between two requests to a server (three seconds for energy-charts, which refuses more), answers kept a while (rates an hour, crypto-assets two minutes, Italy's files three hours). CoinGecko asks to be credited: every answer of it says "Data provided by CoinGecko". The licence of E-Control's interface is not stated on its pages; it is called a public interface.
- **No source run from Russia or China.**

## Shares: the key

Alpha Vantage gives a free key ("it takes fewer than 20 seconds", it says): <https://www.alphavantage.co/support/#api-key>. Put it in `~/.config/janas/prices.conf`:

```
alphavantage_key = YOUR_KEY
```

or in `JANAS_ALPHAVANTAGE_KEY`. The free key allows 25 requests a day (Alpha Vantage lifts the limit for verified open-source projects); a question costs one or two. Without a key, `prices_stock` answers how to get one. The prices may be delayed.

## Running it

```sh
janas-prices [--config FILE]
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients:

```sh
claude mcp add prices -- /path/to/janas-prices
```

```json
{"mcpServers": {"prices": {"command": "/path/to/janas-prices"}}}
```

## What has been run

2 October 2026, over the protocol, in the release and AddressSanitizer builds, without a key of Alpha Vantage: 100 euros in dollars, pounds and francs, dollars into euros and yen on 15 January 2026, pounds into euros and dollars; Bitcoin, Solana in dollars, Dogecoin, a crypto-asset that does not exist; inflation in Italy, Greece, the euro area (Eurostat), the United States and Japan (the World Bank); the rates of the ECB, the Fed, the Bank of England and the SNB; electricity in Milan, Berlin (tomorrow) and Paris, refused for Palermo (not open data); diesel in Trapani, Madrid and Vienna, petrol in Paris, Barcelona and where the user is (Palermo), LPG in Milan and Lyon, methane near Trapani (none within 15 km) and in Paris (not in France's data), LPG in Graz (not in Austria's), diesel in London (not covered); a share without a key (the answer says how to get one). The quotes of Alpha Vantage have been read only with its public demo key (IBM).

In `janas-chat` with Qwen3.6-35B-A3B, the same day, in Italian: 100 euros in dollars and pounds, Bitcoin now, inflation in Italy, the cheapest diesel near the user (Palermo, by GeoJS), electricity in Milan today, the ECB's rates, Apple's share without a key (the model told how to get one). The first run showed a station's price of two months before as the cheapest (prices older than eight days are now left out, and counted), "rates" translated as speeds, and the sources' names left in English; run again, the answers came whole in Italian.
