/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_prices_parse.c - janas-prices' reading of its sources' answers,
 * without the network, and the layouts filled with what was read: the
 * ECB's rates (Frankfurter's and its own file), CoinGecko's search and
 * price, Coinbase's, Alpha Vantage's quote, search and limit, Eurostat's
 * and the World Bank's inflation, the rates of the ECB, the Bank of
 * England, the SNB and the New York Fed, energy-charts' prices (open and
 * not), and the fuel of Italy, France, Spain and Austria, as they came
 * on 2 October 2026 (cut to a few items).
 * The sources belong to the program, so they are compiled in here.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/template.h"
#include "services/prices/data.c"
#include "services/prices/layouts.c"
#include "services/prices/read.c"
#include "services/prices/read_fuel.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char S_FRANKFURTER[] =
    "{\"amount\":1.0,\"base\":\"EUR\",\"date\":\"2026-10-02\",\"rates\":{\""
    "CHF\":0.9279,\"GBP\":0.85033,\"USD\":1.1225}}";
static const char S_ECB_XML[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<gesmes:Envelope xmlns:ges"
    "mes=\"http://www.gesmes.org/xml/2002-08-01\" xmlns=\"http://www.ecb.in"
    "t/vocabulary/2002-08-01/eurofxref\">\n\t<gesmes:subject>Reference rate"
    "s</gesmes:subject>\n\t<gesmes:Sender>\n\t\t<gesmes:name>European Centr"
    "al Bank</gesmes:name>\n\t</gesmes:Sender>\n\t<Cube>\n\t\t<Cube time='2"
    "026-10-02'>\n\t\t\t<Cube currency='USD' rate='1.1225'/>\n\t\t\t</Cube>"
    "</Cube></gesmes:Envelope>";
static const char S_CG_SEARCH[] =
    "{\"coins\": [{\"id\": \"solana\", \"name\": \"Solana\", \"api_symbol\""
    ": \"solana\", \"symbol\": \"SOL\", \"market_cap_rank\": 7}, {\"id\": "
    "\"solana-name-service\", \"name\": \"Solana Name Service\", \"api_symb"
    "ol\": \"solana-name-service\", \"symbol\": \"SNS\", \"market_cap_rank"
    "\": 2510}, {\"id\": \"green-satoshi-token\", \"name\": \"STEPN Green S"
    "atoshi Token on Solana\", \"api_symbol\": \"green-satoshi-token\", \"s"
    "ymbol\": \"GST-SOL\", \"market_cap_rank\": 2833}]}";
static const char S_CG_PRICE[] =
    "{\"solana\":{\"eur\":107.62,\"eur_market_cap\":63316909208.14257,\"eur"
    "_24h_change\":3.607680380747676,\"last_updated_at\":1790952650}}";
static const char S_COINBASE[] =
    "{\"data\":{\"amount\":\"76832.01\",\"base\":\"BTC\",\"currency\":\"EUR"
    "\"}}";
static const char S_AV_QUOTE[] =
    "{\"Global Quote\": {\"01. symbol\": \"IBM\", \"02. open\": \"230.7600"
    "\", \"03. high\": \"234.2700\", \"04. low\": \"224.5000\", \"05. price"
    "\": \"225.6200\", \"06. volume\": \"8884106\", \"07. latest trading da"
    "y\": \"2026-10-01\", \"08. previous close\": \"219.9300\", \"09. chang"
    "e\": \"5.6900\", \"10. change percent\": \"2.5872%\"}}";
static const char S_AV_SEARCH[] =
    "{\"bestMatches\": [{\"1. symbol\": \"TSCO.LON\", \"2. name\": \"Tesco "
    "PLC\", \"3. type\": \"Equity\", \"4. region\": \"United Kingdom\", \"5"
    ". marketOpen\": \"08:00\", \"6. marketClose\": \"16:30\", \"7. timezon"
    "e\": \"UTC+01\", \"8. currency\": \"GBX\", \"9. matchScore\": \"0.7273"
    "\"}, {\"1. symbol\": \"TSCDF\", \"2. name\": \"Tesco plc\", \"3. type"
    "\": \"Equity\", \"4. region\": \"United States\", \"5. marketOpen\": "
    "\"09:30\", \"6. marketClose\": \"16:00\", \"7. timezone\": \"UTC-04\","
    " \"8. currency\": \"USD\", \"9. matchScore\": \"0.7143\"}]}";
static const char S_AV_LIMIT[] =
    "{\"Information\": \"The **demo** API key is for demo purposes only. Pl"
    "ease claim your free API key at (https://www.alphavantage.co/support/#"
    "api-key) to explore our full API offerings. It takes fewer than 20 sec"
    "onds.\"}";
static const char S_EUROSTAT[] =
    "{\"version\": \"2.0\", \"class\": \"dataset\", \"label\": \"Harmonised"
    " index of consumer prices (HICP) - ECOICOP ver.2 - indices and rates o"
    "f change, monthly data\", \"source\": \"ESTAT\", \"updated\": \"2026-1"
    "0-02T11:00:00+0200\", \"value\": {\"0\": 2.2, \"1\": 2.1, \"2\": 2.1, "
    "\"3\": 2.0, \"4\": 1.7, \"5\": 1.9, \"6\": 2.6, \"7\": 3.0, \"8\": 3.2"
    ", \"9\": 2.8, \"10\": 2.9, \"11\": 3.2, \"12\": 3.8}, \"status\": {\"1"
    "2\": \"e\"}, \"id\": [\"freq\", \"unit\", \"coicop18\", \"geo\", \"tim"
    "e\"], \"size\": [1, 1, 1, 1, 13], \"dimension\": {\"freq\": {\"label\""
    ": \"Time frequency\", \"category\": {\"index\": {\"M\": 0}, \"label\":"
    " {\"M\": \"Monthly\"}}}, \"unit\": {\"label\": \"Unit of measure\", \""
    "category\": {\"index\": {\"RCH_A\": 0}, \"label\": {\"RCH_A\": \"Annua"
    "l rate of change\"}}}, \"coicop18\": {\"label\": \"Classification of i"
    "ndividual consumption by purpose (COICOP) - 2018\", \"category\": {\"i"
    "ndex\": {\"TOTAL\": 0}, \"label\": {\"TOTAL\": \"Total\"}}}, \"geo\": "
    "{\"label\": \"Geopolitical entity (reporting)\", \"category\": {\"inde"
    "x\": {\"EA\": 0}, \"label\": {\"EA\": \"Euro area (EA11-1999, EA12-200"
    "1, EA13-2007, EA15-2008, EA16-2009, EA17-2011, EA18-2014, EA19-2015, E"
    "A20-2023, EA21-2026)\"}}}, \"time\": {\"label\": \"Time\", \"category"
    "\": {\"index\": {\"2025-09\": 0, \"2025-10\": 1, \"2025-11\": 2, \"202"
    "5-12\": 3, \"2026-01\": 4, \"2026-02\": 5, \"2026-03\": 6, \"2026-04\""
    ": 7, \"2026-05\": 8, \"2026-06\": 9, \"2026-07\": 10, \"2026-08\": 11,"
    " \"2026-09\": 12}, \"label\": {\"2025-09\": \"2025-09\", \"2025-10\": "
    "\"2025-10\", \"2025-11\": \"2025-11\", \"2025-12\": \"2025-12\", \"202"
    "6-01\": \"2026-01\", \"2026-02\": \"2026-02\", \"2026-03\": \"2026-03"
    "\", \"2026-04\": \"2026-04\", \"2026-05\": \"2026-05\", \"2026-06\": "
    "\"2026-06\", \"2026-07\": \"2026-07\", \"2026-08\": \"2026-08\", \"202"
    "6-09\": \"2026-09\"}}}}}";
static const char S_WORLDBANK[] =
    "[{\"page\":1,\"pages\":1,\"per_page\":50,\"total\":3,\"sourceid\":\"2"
    "\",\"lastupdated\":\"2026-07-13\"},[{\"indicator\":{\"id\":\"FP.CPI.TO"
    "TL.ZG\",\"value\":\"Inflation, consumer prices (annual %)\"},\"country"
    "\":{\"id\":\"GB\",\"value\":\"United Kingdom\"},\"countryiso3code\":\""
    "GBR\",\"date\":\"2025\",\"value\":3.88306881626,\"unit\":\"\",\"obs_st"
    "atus\":\"\",\"decimal\":1},{\"indicator\":{\"id\":\"FP.CPI.TOTL.ZG\","
    "\"value\":\"Inflation, consumer prices (annual %)\"},\"country\":{\"id"
    "\":\"JP\",\"value\":\"Japan\"},\"countryiso3code\":\"JPN\",\"date\":\""
    "2025\",\"value\":3.17253034260248,\"unit\":\"\",\"obs_status\":\"\",\""
    "decimal\":1},{\"indicator\":{\"id\":\"FP.CPI.TOTL.ZG\",\"value\":\"Inf"
    "lation, consumer prices (annual %)\"},\"country\":{\"id\":\"US\",\"val"
    "ue\":\"United States\"},\"countryiso3code\":\"USA\",\"date\":\"2025\","
    "\"value\":null,\"unit\":\"\",\"obs_status\":\"\",\"decimal\":1}]]";
static const char S_ECB_CSV[] =
    "KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DA"
    "TA_TYPE_FM,TIME_PERIOD,OBS_VALUE,OBS_STATUS,OBS_CONF,OBS_PRE_BREAK,OBS"
    "_COM,TIME_FORMAT,BREAKS,COLLECTION,COMPILING_ORG,DISS_ORG,DOM_SER_IDS,"
    "FM_CONTRACT_TIME,FM_COUPON_RATE,FM_IDENTIFIER,FM_LOT_SIZE,FM_MATURITY,"
    "FM_OUTS_AMOUNT,FM_PUT_CALL,FM_STRIKE_PRICE,PUBL_MU,PUBL_PUBLIC,UNIT_IN"
    "DEX_BASE,COMPILATION,COVERAGE,DECIMALS,SOURCE_AGENCY,SOURCE_PUB,TITLE,"
    "TITLE_COMPL,UNIT,UNIT_MULT\nFM.D.U2.EUR.4F.KR.DFR.LEV,D,U2,EUR,4F,KR,D"
    "FR,LEV,2026-10-02,2.5,A,F,,,P1D,,E,4F0,,,,,,,,,,,,,,,,7,,,Deposit faci"
    "lity - date of changes (raw data) - Level,\"Euro area (changing compos"
    "ition) - Key interest rate - Deposit facility - date of changes (raw d"
    "ata) - Level - Euro, provided by ECB\",PCPA,0\n";
static const char S_BOE[] =
    "DATE,IUDBEDR\n01 Sep 2026,3.75\n02 Sep 2026,3.75\n03 Sep 2026,3.75\n04"
    " Sep 2026,3.75\n07 Sep 2026,3.75\n08 Sep 2026,3.75\n09 Sep 2026,3.75\n"
    "10 Sep 2026,3.75\n11 Sep 2026,3.75\n14 Sep 2026,3.75\n15 Sep 2026,3.75"
    "\n16 Sep 2026,3.75\n17 Sep 2026,3.75\n18 Sep 2026,3.75\n21 Sep 2026,3."
    "75\n22 Sep 2026,3.75\n23 Sep 2026,3.75\n24 Sep 2026,3.75\n25 Sep 2026,"
    "3.75\n28 Sep 2026,3.75\n29 Sep 2026,3.75\n30 Sep 2026,3.75\n01 Oct 202"
    "6,3.75\n";
static const char S_SNB[] =
    "{\"timeseries\": [{\"header\": [{\"dim\": \"Overview\", \"dimItem\": "
    "\"SNB policy rate\"}], \"metadata\": {\"key\": \"EPB@SNB.snbgwdzid{LZ}"
    "\", \"frequency\": \"P1D_L\", \"scale\": \"\", \"unit\": \"In percent"
    "\"}, \"values\": [{\"date\": \"2026-09-23\", \"value\": 0.0}, {\"date"
    "\": \"2026-09-24\", \"value\": 0.0}, {\"date\": \"2026-09-25\", \"valu"
    "e\": 0.0}]}]}";
static const char S_NYFED[] =
    "{ \"refRates\": [ { \"effectiveDate\": \"2026-10-01\", \"type\": \"EFF"
    "R\" ,\"percentRate\": 3.88 ,\"percentPercentile1\": 3.85 ,\"percentPer"
    "centile25\": 3.87 ,\"percentPercentile75\": 3.88 ,\"percentPercentile9"
    "9\": 3.89 ,\"targetRateFrom\": 3.75 ,\"targetRateTo\": 4.00 ,\"volumeI"
    "nBillions\": 120 ,\"revisionIndicator\": \"\" } ] }";
static const char S_POWER[] =
    "{\"license_info\": \"CC BY 4.0 (creativecommons.org/licenses/by/4.0) f"
    "rom Bundesnetzagentur | SMARD.de\", \"unix_seconds\": [1790892000, 179"
    "0892900, 1790893800, 1790894700, 1790895600, 1790896500, 1790897400, 1"
    "790898300], \"price\": [185.11, 208.09, 208.09, 196.89, 203.55, 185.0,"
    " 181.75, 183.9], \"unit\": \"EUR / MWh\", \"deprecated\": false}";
static const char S_POWER_CLOSED[] =
    "{\"license_info\": \"The data provided herein is for private and inter"
    "nal use only.\", \"unix_seconds\": [1790892000], \"price\": [100.0], "
    "\"unit\": \"EUR / MWh\"}";
static const char S_IT_REG[] =
    "Estrazione del 2026-10-01\nidImpianto|Gestore|Bandiera|Tipo Impianto|N"
    "ome Impianto|Indirizzo|Comune|Provincia|Latitudine|Longitudine\n18062|"
    "SCHIFANO ALESSANDRO & TAGLIAVIA DAVIDE - SOCIETA' IN NOME COLLE TTIVO|"
    "Api-Ip|Stradale|STAZIONE IP SCHIFANO & TAGLIAVIA|LUNGOMARE DANTE ALIGH"
    "IERI  91100|TRAPANI|TP|38.021958273519594|12.519967354355135\n29846|MA"
    "GGIO GIACOMA|Pompe Bianche|Stradale|pompa bianca|VIA MARSALA 72 91100|"
    "TRAPANI|TP|38.018249806100236|12.532137080426082\n38672|PROTEC S.R.L.|"
    "Nobile Oil|Stradale|PROTEC TP|PIAZZA STAZIONE SNC 91100|TRAPANI|TP|38."
    "01674157298534|12.518189017888517\n45503|EOS SERVICES S.R.L. A SOCIO U"
    "NICO|Q8|Stradale|TP063|VIA MARSALA ANGOLO VIA VESPRI SNC 91100|TRAPANI"
    "|TP|38.01920496451889|12.530868220888578\n41498|GP SERVICE DI PECORELL"
    "A GASPARE & C. SAS|Pompe Bianche|Stradale|GP PETROLI|PIAZZALE LAZZARET"
    "TO SNC 91100|TRAPANI|TP|38.015526026835126|12.49569296836853\n";
static const char S_IT_PRICES[] =
    "Estrazione del 2026-10-01\nidImpianto|descCarburante|prezzo|isSelf|dtC"
    "omu\n18062|Gasolio|2.460|0|30/09/2026 17:10:38\n18062|Gasolio|2.230|1|"
    "30/09/2026 17:10:38\n29846|Gasolio|2.369|1|29/09/2026 18:42:12\n38672|"
    "Gasolio|2.099|0|29/09/2026 08:29:00\n45503|Gasolio|2.699|0|29/09/2026 "
    "07:48:05\n45503|Gasolio|2.399|1|29/09/2026 07:48:05\n41498|Gasolio|2.4"
    "00|0|30/09/2026 21:31:09\n18062|Benzina|2.260|0|30/09/2026 17:10:37\n";
static const char S_FR[] =
    "{\"total_count\": 7, \"results\": [{\"id\": 75013025, \"latitude\": \""
    "4883200\", \"longitude\": \"235900\", \"cp\": \"75013\", \"pop\": \"R"
    "\", \"adresse\": \"181, BOULEVARD VINCENT AURIOL\", \"ville\": \"Paris"
    "\", \"rupture\": \"[{\\\"@nom\\\": \\\"SP95\\\", \\\"@id\\\": \\\"2\\"
    "\", \\\"@debut\\\": \\\"2023-01-30 13:35:10\\\", \\\"@fin\\\": \\\"\\"
    "\", \\\"@type\\\": \\\"definitive\\\"}, {\\\"@nom\\\": \\\"E85\\\", \\"
    "\"@id\\\": \\\"3\\\", \\\"@debut\\\": \\\"2023-01-30 13:35:10\\\", \\"
    "\"@fin\\\": \\\"\\\", \\\"@type\\\": \\\"definitive\\\"}]\", \"geom\":"
    " {\"lon\": 2.359, \"lat\": 48.832}, \"gazole_maj\": \"2026-10-02T09:21"
    ":33+00:00\", \"gazole_prix\": 2.25, \"sp95_maj\": null, \"sp95_prix\":"
    " null, \"e85_maj\": null, \"e85_prix\": null, \"gplc_maj\": \"2026-10-"
    "02T09:21:33+00:00\", \"gplc_prix\": 0.984, \"e10_maj\": \"2026-10-02T0"
    "9:21:33+00:00\", \"e10_prix\": 1.99, \"sp98_maj\": \"2026-10-02T09:21:"
    "33+00:00\", \"sp98_prix\": 1.99, \"e10_rupture_debut\": null, \"e10_ru"
    "pture_type\": null, \"sp98_rupture_debut\": null, \"sp98_rupture_type"
    "\": null, \"sp95_rupture_debut\": \"2023-01-30T13:35:10+00:00\", \"sp9"
    "5_rupture_type\": \"definitive\", \"e85_rupture_debut\": \"2023-01-30T"
    "13:35:10+00:00\", \"e85_rupture_type\": \"definitive\", \"gplc_rupture"
    "_debut\": null, \"gplc_rupture_type\": null, \"gazole_rupture_debut\":"
    " null, \"gazole_rupture_type\": null, \"carburants_rupture_temporaire"
    "\": null, \"carburants_rupture_definitive\": \"SP95;E85\", \"horaires_"
    "automate_24_24\": \"Non\", \"departement\": \"Paris\", \"code_departem"
    "ent\": \"75\", \"region\": \"Île-de-France\", \"code_region\": \"11\""
    "}, {\"id\": 75013024, \"latitude\": \"4883500\", \"longitude\": \"2358"
    "00\", \"cp\": \"75013\", \"pop\": \"R\", \"adresse\": \"114 BD DE L HO"
    "PITAL\", \"ville\": \"Paris\", \"rupture\": \"[{\\\"@nom\\\": \\\"GPLc"
    "\\\", \\\"@id\\\": \\\"4\\\", \\\"@debut\\\": \\\"2020-03-04 19:45:31"
    "\\\", \\\"@fin\\\": \\\"\\\", \\\"@type\\\": \\\"definitive\\\"}, {\\"
    "\"@nom\\\": \\\"SP95\\\", \\\"@id\\\": \\\"2\\\", \\\"@debut\\\": \\\""
    "2025-10-28 06:11:04\\\", \\\"@fin\\\": \\\"\\\", \\\"@type\\\": \\\"de"
    "finitive\\\"}, {\\\"@nom\\\": \\\"SP98\\\", \\\"@id\\\": \\\"6\\\", \\"
    "\"@debut\\\": \\\"2026-10-02 14:03:40\\\", \\\"@fin\\\": \\\"\\\", \\"
    "\"@type\\\": \\\"temporaire\\\"}]\", \"geom\": {\"lon\": 2.358, \"lat"
    "\": 48.835}, \"gazole_maj\": \"2026-10-02T00:01:00+00:00\", \"gazole_p"
    "rix\": 2.25, \"sp95_maj\": null, \"sp95_prix\": null, \"e85_maj\": \"2"
    "026-10-02T00:01:00+00:00\", \"e85_prix\": 0.919, \"gplc_maj\": null, "
    "\"gplc_prix\": null, \"e10_maj\": \"2026-10-02T00:01:00+00:00\", \"e10"
    "_prix\": 1.99, \"sp98_maj\": null, \"sp98_prix\": null, \"e10_rupture_"
    "debut\": null, \"e10_rupture_type\": null, \"sp98_rupture_debut\": \"2"
    "026-10-02T14:03:40+00:00\", \"sp98_rupture_type\": \"temporaire\", \"s"
    "p95_rupture_debut\": \"2025-10-28T06:11:04+00:00\", \"sp95_rupture_typ"
    "e\": \"definitive\", \"e85_rupture_debut\": null, \"e85_rupture_type\""
    ": null, \"gplc_rupture_debut\": \"2020-03-04T19:45:31+00:00\", \"gplc_"
    "rupture_type\": \"definitive\", \"gazole_rupture_debut\": null, \"gazo"
    "le_rupture_type\": null, \"carburants_rupture_temporaire\": \"SP98\", "
    "\"carburants_rupture_definitive\": \"SP95;GPLc\", \"horaires_automate_"
    "24_24\": \"Non\", \"departement\": \"Paris\", \"code_departement\": \""
    "75\", \"region\": \"Île-de-France\", \"code_region\": \"11\"}, {\"id"
    "\": 75012010, \"latitude\": \"4884200\", \"longitude\": \"237300\", \""
    "cp\": \"75012\", \"pop\": \"R\", \"adresse\": \"36 Quai de la Rapée\""
    ", \"ville\": \"Paris\", \"rupture\": \"[{\\\"@nom\\\": \\\"SP95\\\", "
    "\\\"@id\\\": \\\"2\\\", \\\"@debut\\\": \\\"2009-07-02 03:27:00\\\", "
    "\\\"@fin\\\": \\\"\\\", \\\"@type\\\": \\\"\\\"}, {\\\"@nom\\\": \\\"E"
    "85\\\", \\\"@id\\\": \\\"3\\\", \\\"@debut\\\": \\\"2023-06-28 10:21:1"
    "8\\\", \\\"@fin\\\": \\\"\\\", \\\"@type\\\": \\\"temporaire\\\"}, {\\"
    "\"@nom\\\": \\\"GPLc\\\", \\\"@id\\\": \\\"4\\\", \\\"@debut\\\": \\\""
    "2023-06-28 10:21:18\\\", \\\"@fin\\\": \\\"\\\", \\\"@type\\\": \\\"te"
    "mporaire\\\"}]\", \"geom\": {\"lon\": 2.373, \"lat\": 48.842}, \"gazol"
    "e_maj\": \"2026-10-02T07:31:00+00:00\", \"gazole_prix\": 2.391, \"sp95"
    "_maj\": null, \"sp95_prix\": null, \"e85_maj\": null, \"e85_prix\": nu"
    "ll, \"gplc_maj\": null, \"gplc_prix\": null, \"e10_maj\": \"2026-10-02"
    "T07:31:00+00:00\", \"e10_prix\": 2.152, \"sp98_maj\": \"2026-10-02T07:"
    "31:00+00:00\", \"sp98_prix\": 2.286, \"e10_rupture_debut\": null, \"e1"
    "0_rupture_type\": null, \"sp98_rupture_debut\": null, \"sp98_rupture_t"
    "ype\": null, \"sp95_rupture_debut\": null, \"sp95_rupture_type\": null"
    ", \"e85_rupture_debut\": \"2023-06-28T10:21:18+00:00\", \"e85_rupture_"
    "type\": \"temporaire\", \"gplc_rupture_debut\": \"2023-06-28T10:21:18+"
    "00:00\", \"gplc_rupture_type\": \"temporaire\", \"gazole_rupture_debut"
    "\": null, \"gazole_rupture_type\": null, \"carburants_rupture_temporai"
    "re\": \"E85;GPLc\", \"carburants_rupture_definitive\": null, \"horaire"
    "s_automate_24_24\": \"Non\", \"departement\": \"Paris\", \"code_depart"
    "ement\": \"75\", \"region\": \"Île-de-France\", \"code_region\": \"11"
    "\"}]}";
static const char S_ES[] =
    "{\"Fecha\": \"02/10/2026 16:48:00\", \"ListaEESSPrecio\": [{\"Direcci"
    "ón\": \"RONDA SEGOVIA, 37\", \"Latitud\": \"40,410250\", \"Localidad"
    "\": \"MADRID\", \"Longitud (WGS84)\": \"-3,717528\", \"Precio Gasoleo "
    "A\": \"1,899\", \"Precio Gasolina 95 E5\": \"1,748\", \"Rótulo\": \"B"
    "ALLENOIL\"}, {\"Dirección\": \"CL PASEO ACACIAS, 8\", \"Latitud\": \""
    "40,404417\", \"Localidad\": \"MADRID\", \"Longitud (WGS84)\": \"-3,705"
    "389\", \"Precio Gasoleo A\": \"2,005\", \"Precio Gasolina 95 E5\": \"1"
    ",845\", \"Rótulo\": \"REPSOL\"}, {\"Dirección\": \"CALLE ALBERTO AGU"
    "ILERA, 9\", \"Latitud\": \"40,429694\", \"Localidad\": \"MADRID\", \"L"
    "ongitud (WGS84)\": \"-3,708389\", \"Precio Gasoleo A\": \"1,979\", \"P"
    "recio Gasolina 95 E5\": \"1,899\", \"Rótulo\": \"REPSOL\"}, {\"Direcc"
    "ión\": \"GLORIETA EMBAJADORES, 0\", \"Latitud\": \"40,405278\", \"Loc"
    "alidad\": \"MADRID\", \"Longitud (WGS84)\": \"-3,703139\", \"Precio Ga"
    "soleo A\": \"1,969\", \"Precio Gasolina 95 E5\": \"1,849\", \"Rótulo"
    "\": \"BLANCA\"}], \"ResultadoConsulta\": \"OK\"}";
static const char S_AT[] =
    "[{\"id\": 1494440, \"name\": \"TMC Werkstatt & Tankstelle\", \"locatio"
    "n\": {\"address\": \"Rechte Wienzeile 43\", \"postalCode\": \"1050\", "
    "\"city\": \"Wien\", \"latitude\": 48.1963966, \"longitude\": 16.358812"
    "}, \"offerInformation\": {\"service\": true, \"selfService\": false, "
    "\"unattended\": false}, \"position\": 1, \"open\": true, \"distance\":"
    " 0.9219608420880954, \"prices\": [{\"fuelType\": \"DIE\", \"amount\": "
    "2.109, \"label\": \"Diesel\"}]}, {\"id\": 34026, \"name\": \"leodiskon"
    "ttankstelle GmbH\", \"location\": {\"address\": \"Seidengasse 12\", \""
    "postalCode\": \"1070\", \"city\": \"Wien\", \"latitude\": 48.2004272, "
    "\"longitude\": 16.3445982}, \"offerInformation\": {\"service\": true, "
    "\"selfService\": false, \"unattended\": false}, \"position\": 2, \"ope"
    "n\": true, \"distance\": 1.8853604072204895, \"prices\": [{\"fuelType"
    "\": \"DIE\", \"amount\": 2.119, \"label\": \"Diesel\"}]}, {\"id\": 147"
    "6471, \"name\": \"Turmöl\", \"location\": {\"address\": \"Margaretens"
    "traße 28\", \"postalCode\": \"1040\", \"city\": \"Wien\", \"latitude"
    "\": 48.19633, \"longitude\": 16.36464}, \"offerInformation\": {\"servi"
    "ce\": false, \"selfService\": true, \"unattended\": false}, \"position"
    "\": 3, \"open\": true, \"distance\": 0.5701115453999446, \"prices\": ["
    "{\"fuelType\": \"DIE\", \"amount\": 2.124, \"label\": \"Diesel\"}]}, {"
    "\"id\": 6441, \"name\": \"Torrefina Gmbh.\", \"location\": {\"address"
    "\": \"Schottenfeldgasse 94\", \"postalCode\": \"1070\", \"city\": \"Wi"
    "en\", \"latitude\": 48.2071479, \"longitude\": 16.343487}, \"offerInfo"
    "rmation\": {\"service\": true, \"selfService\": false, \"unattended\":"
    " false}, \"position\": 4, \"open\": true, \"distance\": 2.121867918313"
    "451, \"prices\": [{\"fuelType\": \"DIE\", \"amount\": 2.157, \"label\""
    ": \"Diesel\"}]}]";

/* The layout filled with the data: the text, or "" when it failed. */
static char text[16384];

static const char *fill(const char *layout, const struct janas_buf *d)
{
    text[0] = 0;
    struct janas_json_doc *doc =
        janas_json_parse(d->p ? d->p : "", d->n, NULL, 0);
    CHECK(doc != NULL, "the data is not JSON:\n%.*s", (int)d->n, d->p);
    if (!doc)
        return text;
    struct janas_buf out = {0};
    char err[200];
    if (janas_tpl_render(layout, strlen(layout), janas_json_root(doc), &out,
                         err, sizeof err) == 0)
        snprintf(text, sizeof text, "%.*s", (int)out.n, out.p);
    else
        CHECK(0, "a layout not filled: %s", err);
    janas_buf_free(&out);
    janas_json_free(doc);
    return text;
}

static void money(void)
{
    struct pr_fx fx;
    char err[300] = "";
    CHECK(pr_read_frankfurter(S_FRANKFURTER, strlen(S_FRANKFURTER), &fx, err,
                              sizeof err) == 0 &&
              !strcmp(fx.base, "EUR") && !strcmp(fx.date, "2026-10-02") &&
              fx.n == 3,
          "Frankfurter: %s", err);
    struct janas_buf d = {0};
    pr_fx_data(&d, 100, "EUR", &fx, "ecb");
    const char *t = fill(PR_FX_LAYOUT, &d);
    CHECK(strstr(t, "100.00 EUR, at the exchange rates of 2 October 2026:\n") &&
              strstr(t, "  112.25 USD (1 EUR = 1.1225 USD)\n") &&
              strstr(t, "For information only"),
          "the rates' layout:\n%s", t);
    janas_buf_free(&d);
    CHECK(pr_read_ecb_xml(S_ECB_XML, strlen(S_ECB_XML), &fx, err, sizeof err) ==
                  0 &&
              fx.n == 1 && !strcmp(fx.r[0].code, "USD") &&
              fabs(fx.r[0].rate - 1.1225) < 1e-9 &&
              !strcmp(fx.date, "2026-10-02"),
          "the ECB's file: %s %d", err, fx.n);

    struct pr_coin c;
    CHECK(pr_read_coin_search(S_CG_SEARCH, strlen(S_CG_SEARCH), "SOL", &c, err,
                              sizeof err) == 0 &&
              !strcmp(c.id, "solana"),
          "CoinGecko's search: %s %s", err, c.id);
    snprintf(c.cur, sizeof c.cur, "EUR");
    CHECK(pr_read_coin_price(S_CG_PRICE, strlen(S_CG_PRICE), &c, err,
                             sizeof err) == 0 &&
              fabs(c.price - 107.62) < 1e-9 && c.change_24h > 3.6 &&
              c.at == 1790952650,
          "CoinGecko's price: %s", err);
    snprintf(c.symbol, sizeof c.symbol, "SOL");
    pr_coin_data(&d, &c, "coingecko");
    t = fill(PR_COIN_LAYOUT, &d);
    CHECK(
        strstr(t, "Solana (SOL): 107.62 EUR, +3.61% in 24 hours; market "
                  "value 63.3 billion EUR") &&
            strstr(
                t,
                "Data provided by CoinGecko (https://www.coingecko.com/en/api)."),
        "the coin's layout:\n%s", t);
    janas_buf_free(&d);
    CHECK(pr_read_coinbase(S_COINBASE, strlen(S_COINBASE), &c, err,
                           sizeof err) == 0 &&
              fabs(c.price - 76832.01) < 1e-6,
          "Coinbase: %s", err);

    struct pr_quote q;
    memset(&q, 0, sizeof q);
    CHECK(pr_read_av_quote(S_AV_QUOTE, strlen(S_AV_QUOTE), &q, err,
                           sizeof err) == 0 &&
              !strcmp(q.symbol, "IBM") && fabs(q.price - 225.62) < 1e-9 &&
              fabs(q.change_pct - 2.5872) < 1e-9 && q.volume == 8884106 &&
              !strcmp(q.day, "2026-10-01"),
          "Alpha Vantage's quote: %s", err);
    snprintf(q.name, sizeof q.name, "IBM");
    pr_quote_data(&d, &q);
    t = fill(PR_QUOTE_LAYOUT, &d);
    CHECK(strstr(t, "IBM (IBM): 225.62 , +5.69 (+2.59%); the trading day of 1 "
                    "October 2026\nopen 230.76, high 234.27, low 224.50, "
                    "previous close 219.93, shares traded 8884106\n"),
          "the quote's layout:\n%s", t);
    janas_buf_free(&d);
    struct pr_quote s;
    CHECK(pr_read_av_search(S_AV_SEARCH, strlen(S_AV_SEARCH), &s, err,
                            sizeof err) == 0 &&
              !strcmp(s.symbol, "TSCO.LON") && !strcmp(s.currency, "GBX"),
          "Alpha Vantage's search: %s", err);
    CHECK(pr_read_av_quote(S_AV_LIMIT, strlen(S_AV_LIMIT), &q, err,
                           sizeof err) != 0 &&
              strstr(err, "Alpha Vantage: The **demo** API key"),
          "Alpha Vantage's refusal said: %s", err);
}

static void macro(void)
{
    struct pr_series s;
    char err[300] = "", country[64] = "";
    CHECK(pr_read_eurostat(S_EUROSTAT, strlen(S_EUROSTAT), &s, err,
                           sizeof err) == 0 &&
              s.n == 13 && !strcmp(s.p[0].when, "2025-09") &&
              fabs(s.p[12].v - 3.8) < 1e-9,
          "Eurostat: %s %d", err, s.n);
    snprintf(s.unit, sizeof s.unit, "%%");
    struct janas_buf d = {0};
    pr_series_data(&d, "inflation", "the euro area", &s, 13, "eurostat",
                   "flash", "");
    const char *t = fill(PR_SERIES_LAYOUT, &d);
    CHECK(strstr(t, "Inflation, the yearly change of consumer prices, the "
                    "euro area:\n  2025-09: 2.2%\n") &&
              strstr(t, "  2026-09: 3.8%\n") &&
              strstr(t, "minimum 1.7% (2026-01), maximum 3.8% (2026-09)"),
          "the inflation's layout:\n%s", t);
    t = fill(PR_SERIES_BRIEF, &d);
    CHECK(strstr(t, "last 3.8% (2026-09)"), "the inflation's brief:\n%s", t);
    janas_buf_free(&d);
    CHECK(pr_read_worldbank(S_WORLDBANK, strlen(S_WORLDBANK), &s, country,
                            sizeof country, err, sizeof err) == 0 &&
              s.n == 1 && !strcmp(country, "United Kingdom"),
          "the World Bank: %s %d %s", err, s.n, country);
    CHECK(pr_read_ecb_csv(S_ECB_CSV, strlen(S_ECB_CSV), &s, err, sizeof err) ==
                  0 &&
              s.n == 1 && !strcmp(s.p[0].when, "2026-10-02") &&
              fabs(s.p[0].v - 2.5) < 1e-9,
          "the ECB's CSV: %s", err);
    CHECK(pr_read_boe_csv(S_BOE, strlen(S_BOE), &s, err, sizeof err) == 0 &&
              !strcmp(s.p[0].when, "2026-09-01") &&
              fabs(s.p[s.n - 1].v - 3.75) < 1e-9,
          "the Bank of England: %s %s", err, s.n ? s.p[0].when : "");
    CHECK(pr_read_snb(S_SNB, strlen(S_SNB), &s, err, sizeof err) == 0 &&
              s.n == 3 && s.p[2].v == 0,
          "the SNB: %s", err);
    double lo, hi;
    CHECK(pr_read_nyfed(S_NYFED, strlen(S_NYFED), &s, &lo, &hi, err,
                        sizeof err) == 0 &&
              fabs(s.p[0].v - 3.88) < 1e-9 && lo == 3.75 && hi == 4.0,
          "the New York Fed: %s", err);
    struct pr_rate r[2] = {{.name = "effr", .when = "2026-10-01", .v = 3.88}};
    pr_rates_data(&d, "fed", r, 1, lo, hi, "nyfed");
    t = fill(PR_RATES_LAYOUT, &d);
    CHECK(strstr(t, "Federal Reserve:\n  effective federal funds rate: 3.88% "
                    "(1 October 2026)\n  the target range of the federal "
                    "funds: 3.75-4.00%\n"),
          "the rates' layout:\n%s", t);
    janas_buf_free(&d);
    char licence[300];
    CHECK(pr_read_power(S_POWER, strlen(S_POWER), "Europe/Rome", &s, licence,
                        sizeof licence, err, sizeof err) == 1 &&
              s.n == 8 && !strcmp(s.p[0].when, "2026-10-02 00:00") &&
              strstr(licence, "CC BY 4.0"),
          "energy-charts, open: %s %d %s", err, s.n, s.n ? s.p[0].when : "");
    CHECK(pr_read_power(S_POWER_CLOSED, strlen(S_POWER_CLOSED), "", &s, licence,
                        sizeof licence, err, sizeof err) == 0 &&
              s.n == 0,
          "energy-charts, not open: no prices kept");
}

static void fuel(void)
{
    struct pr_fuel f;
    char err[300] = "";
    struct pr_fuel_ask a = {.now = 1790942400,
                            .lat = 38.0174,
                            .lon = 12.5160,
                            .radius_km = 2,
                            .kind = PR_DIESEL};
    CHECK(pr_read_fuel_it(S_IT_REG, strlen(S_IT_REG), S_IT_PRICES,
                          strlen(S_IT_PRICES), &a, &f, err, sizeof err) == 0 &&
              f.found == 5 && f.n == 5,
          "Italy: %s %d", err, f.found);
    CHECK(f.s[0].price == 2.099 && f.s[0].self == 0 &&
              !strcmp(f.s[0].name, "Nobile Oil") && f.s[1].price == 2.230 &&
              f.s[1].self == 1 && !strcmp(f.s[1].name, "Api-Ip") &&
              !strcmp(f.s[2].name, "MAGGIO GIACOMA") &&
              fabs(f.average - 2.2994) < 1e-9,
          "Italy's stations: %s %.3f %d, %s, %s, %.4f", f.s[0].name,
          f.s[1].price, f.s[1].self, f.s[1].name, f.s[2].name, f.average);
    struct pr_place p = {
        .name = "Trapani", .country = "Italia", .lat = 38.0174, .lon = 12.5160};
    struct janas_buf d = {0};
    pr_fuel_data(&d, &p, "diesel", 2, &f, "mimit");
    const char *t = fill(PR_FUEL_LAYOUT, &d);
    CHECK(strstr(t, "Diesel near Trapani, Italia: stations within 2 km, 5; "
                    "the cheapest 2.099 €/l, the average 2.299 €/l\n") &&
              strstr(t, "1. 2.099 €/l (served): Nobile Oil, PIAZZA STAZIONE "
                        "SNC 91100, TRAPANI (") &&
              strstr(t, "2. 2.230 €/l (self-service): Api-Ip,"),
          "the fuel's layout:\n%s", t);
    janas_buf_free(&d);

    a = (struct pr_fuel_ask){.now = 1790942400,
                             .lat = 48.8566,
                             .lon = 2.3522,
                             .radius_km = 3,
                             .kind = PR_DIESEL};
    CHECK(pr_read_fuel_fr(S_FR, strlen(S_FR), &a, &f, err, sizeof err) == 0 &&
              f.n >= 1 && f.found == 7 && isnan(f.average) &&
              !strncmp(f.s[0].updated, "2026-10-02 ", 11),
          "France: %s %d %d %s", err, f.n, f.found, f.s[0].updated);
    a.kind = PR_CNG;
    CHECK(pr_read_fuel_fr(S_FR, strlen(S_FR), &a, &f, err, sizeof err) != 0,
          "France has no methane");
    a = (struct pr_fuel_ask){.now = 1790942400,
                             .lat = 40.4168,
                             .lon = -3.7038,
                             .radius_km = 2,
                             .kind = PR_DIESEL};
    CHECK(pr_read_fuel_es(S_ES, strlen(S_ES), &a, &f, err, sizeof err) == 0 &&
              f.n >= 1 && f.s[0].price > 1 && f.s[0].price < 3 &&
              !strncmp(f.s[0].updated, "2026-10-02 ", 11),
          "Spain: %s %d", err, f.n);
    a = (struct pr_fuel_ask){.now = 1790942400,
                             .lat = 48.2,
                             .lon = 16.37,
                             .radius_km = 5,
                             .kind = PR_DIESEL};
    CHECK(pr_read_fuel_at(S_AT, strlen(S_AT), &a, &f, err, sizeof err) == 0 &&
              f.n >= 1 && fabs(f.s[0].price - 2.109) < 1e-9 && f.s[0].self == 0,
          "Austria: %s %d", err, f.n);
}

/* a price told over PR_STALE_DAYS days before is left out, in both
   ways the sources write a time */
static void stale(void)
{
    struct pr_station st[3] = {
        {.name = "old", .price = 1.5, .updated = "16/09/2026 08:00:00"},
        {.name = "new", .price = 1.9, .updated = "2026-10-01 10:00"},
        {.name = "undated", .price = 1.8}};
    struct pr_fuel f;
    memset(&f, 0, sizeof f);
    pr_fuel_rank(&f, st, 3, 1790942400);
    CHECK(f.found == 2 && f.stale == 1 && !strcmp(f.s[0].name, "undated") &&
              f.cheapest == 1.8,
          "stale prices: %d found, %d left out, first %s", f.found, f.stale,
          f.s[0].name);
}

int main(void)
{
    setenv("TZ", "UTC", 1);
    money();
    macro();
    fuel();
    stale();
    if (failures) {
        printf("test_prices_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_prices_parse: ok\n");
    return 0;
}
