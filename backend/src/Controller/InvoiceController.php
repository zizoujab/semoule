<?php

namespace App\Controller;

use ApiPlatform\Metadata\ApiResource;
use ApiPlatform\Metadata\Post;
use App\Service\InvoiceService;
use Symfony\Component\HttpFoundation\BinaryFileResponse;
use Symfony\Component\HttpFoundation\File\Exception\FileException;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpKernel\Attribute\AsController;
use Symfony\Component\Routing\Annotation\Route;
use Symfony\Contracts\HttpClient\Exception\TransportExceptionInterface;
use Symfony\Contracts\HttpClient\HttpClientInterface;

#[ApiResource(
    operations: [
        new Post(
            controller: InvoiceController::class,
            name: 'post invoice',
        )
    ],

)]
#[AsController]
class InvoiceController
{
    private HttpClientInterface $httpClient;

    // Injecting the HttpClient to send requests to the /ocr_old API
    public function __construct(HttpClientInterface $httpClient)
    {
        $this->httpClient = $httpClient;
    }

    #[Route('/api/invoice/ocr', name: 'invoice_ocr', methods: ['POST'])]
    public function ocr(Request $request, InvoiceService $invoiceService): JsonResponse
    {
        // Check if the image is present
        $image = $request->files->get('image');
        if (!$image) {
            return new JsonResponse(['error' => 'No image provided'], 400);
        }

        // Send the image to the OCR endpoint (Assuming it is running on http://localhost:5000/ocr)
        try {
            // Open the image and send it to the OCR service
            $response = $this->httpClient->request('POST', 'http://ocr:5000/ocr', [
                'body' => [
                    'image' => fopen($image->getPathname(), 'r')
                ]
            ]);

            // Parse the response from OCR
            $ocrData = $response->toArray();
            return new JsonResponse([
                'text' => $ocrData['extracted_text'],
                'amounts' => $invoiceService->extractData($ocrData['extracted_text'])
            ]);

        } catch (FileException $e) {
            return new JsonResponse(['error' => 'Error processing the image: ' . $e->getMessage()], 500);
        } catch (\Exception $e) {
            return new JsonResponse(['error' => 'OCR service error: ' . $e->getMessage()], 500);
        }

    }

    #[Route('/api/invoice/ml', name: 'invoice_ml', methods: ['POST'])]
    public function ml(Request $request, InvoiceService $invoiceService, HttpClientInterface $httpClient)
    {
        // Check if the image is present
        $image = $request->files->get('image');
        if (!$image) {
            return new JsonResponse(['error' => 'No image provided'], 400);
        }
        // Send the image to the OCR endpoint (Assuming it is running on http://localhost:5000/ocr)
        try {
            // Open the image and send it to the OCR service
            $response = $this->httpClient->request('POST', 'http://ocr:5000/ml', [
                'body' => [
                            'image' => fopen($image->getPathname(), 'rb'),
                            'filename' => $image->getClientOriginalName(),
                            'headers' => ['Content-Type' => $image->getMimeType()],
                ]
            ]);

            // Parse the response from ML
            $responseArray = $response->toArray();
//            return new JsonResponse($responseArray);
            $annotatedImage = $httpClient->request('GET', sprintf('http://ocr:5000/%s', $responseArray['file_uri']));
            file_put_contents($responseArray['file_name'], $annotatedImage->getContent());
            return new BinaryFileResponse($responseArray['file_name']);
//            return new JsonResponse([
//                'text' => $ocrData['extracted_text'],
//                'amounts' => $invoiceService->extractData($ocrData['extracted_text'])
//            ]);

        } catch (FileException $e) {
            return new JsonResponse(['error' => 'Error processing the image: ' . $e->getMessage()], 500);
        } catch (TransportExceptionInterface $e) {
            return new JsonResponse(['error' => 'Something went wrong while trying to download the annotated image' . $e->getMessage()], 500);

        }
        catch (\Exception $e) {
            return new JsonResponse(['error' => 'OCR service error: ' . $e->getMessage()], 500);
        }

    }

}