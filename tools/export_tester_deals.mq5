//+------------------------------------------------------------------+
//| MQL5 Trade Exporter — run inside MT5 Strategy Tester              |
//|                                                                    |
//| 1. Open MT5 Strategy Tester (View → Strategy Tester)              |
//| 2. Configure your EA + symbol + timeframe                         |
//| 3. Run a single backtest                                          |
//| 4. Place this file in MQL5/Scripts/                               |
//| 5. In MetaEditor, attach it to a chart as a Service OR run via    |
//|    Tools → Script Editor → compile → drag onto Tester chart       |
//| 6. After the test, the script writes tester_trades.csv to         |
//|    MQL5/Files/                                                    |
//|                                                                    |
//| Note: Strategy Tester trade history is accessed via               |
//|   HistorySelect(start, end) + HistoryDealsTotal() + HistoryDealGetTicket() |
//| on the **deals** (not orders). Each position has 2 deals: entry   |
//| and exit. We pair them into round-trip trades.                    |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input datetime "From" = D'2022.01.01 00:00:00';
input datetime "To"   = D'2024.12.31 23:59:59';
input int     MagicNumberFilter = 333777;  // EA's magic number
input string  OutputFileName = "tester_trades.csv";

void OnStart()
{
    if(!HistorySelect(_From, _To))
    {
        Print("HistorySelect failed: ", GetLastError());
        return;
    }
    int total = HistoryDealsTotal();
    Print("Total deals in window: ", total);

    // Collect deals grouped by position_id
    // Each entry deal sets POSITION_ID; exit deal sets POSITION_ID too.
    string rows = "";
    rows += "deal_time,deal_type,position_id,order_id,symbol,volume,price,profit,swap,commission,magic,entry\n";

    int deals_in = 0;
    int deals_out = 0;
    for(int i = 0; i < total; i++)
    {
        ulong ticket = HistoryDealGetTicket(i);
        if(ticket == 0) continue;
        datetime dt = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
        long   type = HistoryDealGetInteger(ticket, DEAL_TYPE);
        long   pos_id = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
        long   order_id = HistoryDealGetInteger(ticket, DEAL_ORDER);
        string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
        double vol = HistoryDealGetDouble(ticket, DEAL_VOLUME);
        double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
        double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
        double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
        double comm = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
        long   magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
        if(MagicNumberFilter > 0 && magic != MagicNumberFilter) continue;

        string entry_flag = (type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL) ? "1" : "0";
        if(type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL) deals_in++;
        if(type == DEAL_TYPE_SELL || type == DEAL_TYPE_BUY) deals_out++;

        rows += StringFormat("%s,%d,%d,%d,%s,%.4f,%.5f,%.2f,%.2f,%.2f,%d,%s\n",
                              TimeToString(dt, TIME_DATE|TIME_MINUTES),
                              type, pos_id, order_id, sym, vol, price,
                              profit, swap, comm, magic, entry_flag);
    }

    string path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + _OutputFileName;
    // Write to common Files dir (works for both tester and chart)
    int h = FileOpen(_OutputFileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
    if(h == INVALID_HANDLE)
    {
        Print("FileOpen failed: ", GetLastError());
        return;
    }
    FileWriteString(h, rows);
    FileClose(h);
    Print("Wrote ", deals_in, " entry + ", deals_out, " exit deals to ", _OutputFileName);
    Print("Also at: ", path);
}