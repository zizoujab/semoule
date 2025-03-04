<?php

namespace App\Service;

class InvoiceService
{

    public function extractData(string $text): array
    {
        $amounts = [];
        preg_match_all('/\d{1,3}\s*,\s*\d{2}/', $text, $amounts);
        return $amounts;
    }

}